import torch
from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states as triton_merge_attn_states
import pytest
from vllm.platforms import current_platform

def merge_attn_states(
    output: torch.Tensor,                       # 最终合并后写回的全局Attention输出矩阵O    [num_tokens,num_head,head_size]
    prefix_output: torch.Tensor,                # 前缀局部输出                             [num_tokens,num_head,head_size]
    prefix_lse: torch.Tensor,                   # 前缀注意力尺度LSE                         [num_head,num_token]
    suffix_output: torch.Tensor,                #后缀局部输出                               [num_tokens,num_head,head_size]
    suffix_lse: torch.Tensor,                   #后缀注意力尺度LSE                          [num_head,num_token]
    output_lse: torch.Tensor | None = None,     # 合并后的全局注意力尺度LSE                 [num_head,num_token]
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

BATCH_NUM_TOKENS = [256,512,613,1024,1536,2048,4096]
NUM_QUERY_HEADS = [16,32,48,64,128]
HEAD_SIZES = [32,48,64,96,128,256]
DTYPES = [torch.float32]

@pytest.mark.parametrize("num_tokens",BATCH_NUM_TOKENS)
@pytest.mark.parametrize("num_query_heads",NUM_QUERY_HEADS)
@pytest.mark.parametrize("head_size",HEAD_SIZES)
@pytest.mark.parametrize("output_dtypes",DTYPES)
@torch.inference_mode()
def test_merge_attn_states(num_tokens:int, num_query_heads:int, head_size:int, output_dtypes:torch.dtype):

    NUM_TOKENS = num_tokens
    NUM_HEADS = num_query_heads
    HEAD_SIZE = head_size

    print(f"\nNUM TOKEN:{NUM_TOKENS}, NUM HEAD:{NUM_HEADS}, HEAD SIZE:{HEAD_SIZE}, DEVICE:{current_platform.device_name}\n")
    print(100*"-")

    prefix_lse = torch.randn((NUM_HEADS,NUM_TOKENS),dtype=torch.float32,device="cuda")
    suffix_lse = torch.randn((NUM_HEADS,NUM_TOKENS),dtype=torch.float32,device="cuda")

    # 生成一些inf值
    prefix_mask = torch.rand(NUM_HEADS,NUM_TOKENS)<0.1
    suffix_mask = torch.rand(NUM_HEADS,NUM_TOKENS)<0.1

    #   有一些位置的前缀和和后缀和可能都为0
    combined_mask = torch.logical_and(prefix_mask,suffix_mask)

    #   消除都为0的位置
    prefix_mask = torch.logical_and(prefix_mask,combined_mask)
    suffix_mask = torch.logical_and(suffix_mask,combined_mask)

    prefix_lse[prefix_mask] = torch.inf
    suffix_lse[suffix_mask] = torch.inf

    output_lse = torch.zeros((NUM_HEADS,NUM_TOKENS),dtype=output_dtypes,device="cuda")
    output = torch.zeros((NUM_TOKENS,NUM_HEADS,HEAD_SIZE),dtype=output_dtypes,device="cuda")

    prefix_output = torch.randn((NUM_TOKENS,NUM_HEADS,HEAD_SIZE),dtype=output_dtypes,device="cuda")
    suffix_output = torch.randn((NUM_TOKENS,NUM_HEADS,HEAD_SIZE),dtype=output_dtypes,device="cuda")

    warmup_times = 10
    repeat_times = 20

    output_lse_torch = output_lse.clone()
    output_torch = output.clone()
    torch_total_time = 0
    start = torch.Event(enable_timing=True)
    end  = torch.Event(enable_timing=True)

    for _ in range(warmup_times):
        merge_attn_states(
            output_torch,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_torch
            )
        
    torch.cuda.synchronize()

    for _ in range(repeat_times):
        start.record()
        merge_attn_states(
            output_torch,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse_torch
            )
        end.record()
        torch.cuda.synchronize()
        torch_total_time+=start.elapsed_time(end)

    torch_avg_time = torch_total_time/repeat_times
    
    triton_output_lse = output_lse.clone()
    triton_output = output.clone()
    triton_total_time = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    for _ in range(warmup_times):
        triton_merge_attn_states(
            triton_output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            triton_output_lse
        )
    
    torch.cuda.synchronize()

    for _ in range(repeat_times):
        start.record()
        triton_merge_attn_states(
            triton_output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            triton_output_lse
        )
        end.record()
        torch.cuda.synchronize()
        triton_total_time+=start.elapsed_time(end)

    triton_avg_time = triton_total_time/repeat_times

    print("-"*100)
    print(f"the torch time is {torch_avg_time:.6f}\n")
    print(f"the triton time is {triton_avg_time:.6f}\n")
    print("-"*100)



