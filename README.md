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

论文里将注意力状态定义注意力输出和注意力尺度的元组：$\begin{bmatrix}
O(\mathcal{J}) \\
\text{LSE}(\mathcal{J})
\end{bmatrix}$那么对于 $\mathcal{T} \cup \mathcal{J}$ 的注意力状态就可以通过引入算子$\oplus$进行计算：

$$
\begin{aligned}
\begin{bmatrix} \mathbf{O}(\mathcal{J} \cup \mathcal{J}) \\ \text{LSE}(\mathcal{J} \cup \mathcal{J}) \end{bmatrix} 
&= \begin{bmatrix} \mathbf{O}(\mathcal{J}) \\ \text{LSE}(\mathcal{J}) \end{bmatrix} 
\oplus 
\begin{bmatrix} \mathbf{O}(\mathcal{J}) \\ \text{LSE}(\mathcal{J}) \end{bmatrix} \\[6pt]
&= \begin{bmatrix} 
\displaystyle\frac{\exp(\text{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J}) + \exp(\text{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J})}{\exp(\text{LSE}(\mathcal{J})) + \exp(\text{LSE}(\mathcal{J}))} \\[12pt]
\log(\exp(\text{LSE}(\mathcal{J})) + \exp(\text{LSE}(\mathcal{J})))
\end{bmatrix}
\end{aligned}
$$

