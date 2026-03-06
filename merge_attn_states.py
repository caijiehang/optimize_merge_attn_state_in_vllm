import torch

from vllm.platforms import current_platform

def merge_attn_states(
    output: torch.Tensor,                       # 最终合并后写回的全局Attention输出矩阵O
    prefix_output: torch.Tensor,                # 前缀局部输出
    prefix_lse: torch.Tensor,                   # 前缀注意力尺度LSE
    suffix_output: torch.Tensor,                #后缀局部输出
    suffix_lse: torch.Tensor,                   #后缀注意力尺度LSE
    output_lse: torch.Tensor | None = None,     # 合并后的全局注意力尺度LSE
) -> None:
    
    def supported_dtype(o:torch.Tensor) -> bool:
        return o.dtype in [torch.float32,torch.bfloat16,torch.float16]
    
    def supported_headdim(o:torch.Tensor) -> bool:
        headdim = o.shape[2]    #   [num_tokens,num_heads,headsize]
        if o.dtype == torch.float32:
            return headdim%4==0
        return headdim%8==0
    
    if(current_platform.is_cuda() and supported_dtype(output) and supported_headdim(output)):
        from vllm._custom_ops import merge_attn_states
        merge_attn_states(output,prefix_output,prefix_lse,suffix_output,suffix_lse,output_lse)

    else:
        from vllm.v1.attention.ops import triton_merge_attn_states as merge_attn_states
        merge_attn_states(output,prefix_output,prefix_lse,suffix_output,suffix_lse,output_lse)
