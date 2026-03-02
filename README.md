# optimize_merge_attn_state_in_vllm
## Merge Attention States简介
[FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/html/2501.01005?_immersive_translate_auto_translate=1)

我们知道，对于《Attention is all you need》里面的 Attention 计算公式为：

$$O = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

假设缩放因子 $\sqrt{d_k}$ 已经被吸收到 $\mathbf{q}$ 或 $\mathbf{k}$ 中，那么对于单个查询向量 $\mathbf{q}$，它与序列中所有 $N$ 个键 $\mathbf{k}_i$ 和值 $\mathbf{v}_i$ 作用后的全局注意力输出 $\mathbf{o}$ 应当为：

$$\mathbf{o} = \sum_{i=1}^N \frac{\exp(\mathbf{q} \cdot \mathbf{k}_i)}{\sum_{j=1}^N \exp(\mathbf{q} \cdot \mathbf{k}_j)} \mathbf{v}_i$$

对于分母的全局归一化项，在标准的 Transformer 中，必须遍历完所有 $N$ 个 Token 才能计算出这个分母，这严重阻碍了完全并行的分块计算。为了实现块并行（Block-Parallel），我们假设把长度为 $N$ 的序列切分为若干个块。以 $\mathcal{J}$ 表示其中某一个块的索引集合（例如，第 1 到 64 个 Token）。在计算该块的局部注意力时，其局部归一化项为：

$$\text{局部归一化项} = \sum_{j \in \mathcal{J}} \exp(\mathbf{q} \cdot \mathbf{k}_j)$$

直接保存局部归一化项在数值上极易导致指数溢出。因此，我们引入 Log-Sum-Exp (LSE) 技巧来保证数值稳定性。我们将块 $\mathcal{J}$ 的注意力尺度（Attention Scale）定义为局部归一化项的对数：

$$\mathbf{LSE}(\mathcal{J}) = \log \sum_{i \in \mathcal{J}} \exp(\mathbf{q} \cdot \mathbf{k}_i)$$

基于此，查询向量 $\mathbf{q}$ 在块 $\mathcal{J}$ 上的局部注意力输出 $\mathbf{O}(\mathcal{J})$ 就可以表示为：

$$\mathbf{O}(\mathcal{J}) = \sum_{i \in \mathcal{J}} \frac{\exp(\mathbf{q} \cdot \mathbf{k}_i)}{\exp(\mathbf{LSE}(\mathcal{J}))} \cdot \mathbf{v}_i$$

同理，对于查询 $\mathbf{q}$ 在另一个块 $\mathcal{T}$ 的局部注意力输出 $\mathbf{O}(\mathcal{T})$ 为

$$\mathbf{O}(\mathcal{T}) = \sum_{j \in \mathcal{T}} \frac{\exp(\mathbf{q} \cdot \mathbf{k}_j)}{\exp(\mathbf{LSE}(\mathcal{T}))} \cdot \mathbf{v}_i$$

论文里将注意力状态定义注意力输出和注意力尺度的元组：
$\begin{bmatrix}
O(\mathcal{J}) \\ 
\text{LSE}(\mathcal{J})
\end{bmatrix}$
那么对于 $\mathcal{T} \cup \mathcal{J}$ 的注意力状态就可以通过引入算子$\oplus$进行计算：

$$
\begin{aligned}
\begin{bmatrix} 
\mathbf{O}(\mathcal{J} \cup \mathcal{T}) \\
\text{LSE}(\mathcal{J} \cup \mathcal{T}) 
\end{bmatrix} 
&= 
\begin{bmatrix} 
\mathbf{O}(\mathcal{J}) \\
\text{LSE}(\mathcal{J}) 
\end{bmatrix} 
\oplus 
\begin{bmatrix}
\mathbf{O}(\mathcal{T}) \\
\text{LSE}(\mathcal{T}) 
\end{bmatrix} \\
&= \begin{bmatrix} 
\displaystyle\frac{\exp(\text{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J}) + \exp(\text{LSE}(\mathcal{T}))\mathbf{O}(\mathcal{T})}{\exp(\text{LSE}(\mathcal{J})) + \exp(\text{LSE}(\mathcal{T}))} \\
\log(\exp(\text{LSE}(\mathcal{J})) + \exp(\text{LSE}(\mathcal{T})))
\end{bmatrix}
\end{aligned}
$$

## benchmark测试
### pytorch

首先我们可以看一下`triton`实现过程中，所需要输入的数据
```python
def merge_attn_states(
    output: torch.Tensor,                   
    prefix_output: torch.Tensor,            
    prefix_lse: torch.Tensor,               
    suffix_output: torch.Tensor,            
    suffix_lse: torch.Tensor,               
    output_lse: torch.Tensor | None = None, 
)
```

`output`为`prefix_output`和`suffix_output`进行`merge`后输出的矩阵，维度为`[NUM_TOKENS, NUM_HEADS, HEAD_SIZE]` \
`prefix_output`与`suffix_output`为`切分token`中的`prefix`和`suffix`计算结果，维度为`[NUM_TOKENS, NUM_HEADS, HEAD_SIZE]`    \
`prefix_lse`与`suffix_lse`为`prefix`和`suffix`中的对数局部归一化值，由于我们在`MHA`中是对不同的头并行`attention`计算，所以该维度为`[NUM_HEADS, NUM_TOKENS]` \
`output_lse`为`prefix_lse`与`suffix_lse`最终合并后的结果，维度同上。

那么根据上述的思路可以有以下的pytorch实现：

```python
def merge_attn_states(
    output: torch.Tensor,                  
    prefix_output: torch.Tensor,           
    prefix_lse: torch.Tensor,              
    suffix_output: torch.Tensor,           
    suffix_lse: torch.Tensor,              
    output_lse: torch.Tensor | None = None,
):
    p_lse = prefix_lse  # [num_head,num_token]
    s_lse = suffix_lse  # [num_head,num_token]

    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf

    max_lse = torch.maximum(p_lse,s_lse)    #   [num_head,num_token]
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse

    p_exp_lse = torch.exp(p_lse)    #   [num_head,num_token]
    s_exp_lse = torch.exp(s_lse)    #   [num_head,num_token]

    out_se = p_exp_lse + s_exp_lse  #   [num_head,num_token]
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse

    p_scale = p_exp_lse/out_se      #   [num_head,num_token]   
    s_scale = s_exp_lse/out_se      #   [num_head,num_token]

    p_scale = torch.transpose(p_scale,0,1).unsqueeze(2) #   [num_token,num_head,1]
    s_scale = torch.transpose(s_scale,0,1).unsqueeze(2) #   [num_token,num_head,1]

    output = prefix_output*p_scale+suffix_output*s_scale

    return output,output_lse
```

这里中间为什么会有

```python
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
```

原因在`triton`源码中有一段话：“`Flash Attention2`和`Flash Attention3`中，面对`token length=0`的情况有不同的返回值，分别为`inf`和`-inf`。”

对于`merge attention state`来说，后续要进行指数运算，如果返回的是`-inf`，那么 $\exp(-\inf) = 0$ ，表示这个`token length = 0` 的切块对`attention`的贡献为0，这是正确的，但是如果是`inf`,则 $\exp(\inf) = \inf$ ，这就将导致数值溢出，计算结果错误，所以在这里加了一层过滤，对`Flash Attention2`的错误进行纠正
