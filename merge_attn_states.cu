#include <optional>
#include "ATen/core/TensorBody.h"
#include "attention_utils.cuh"
#include "c10/util/Exception.h"
#include "dtype_float32.cuh"
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>


namespace vllm {
    template <typename scalar_t , const uint NUM_THREADS>
    __global__ void merge_attn_state_kernel(
        scalar_t* output, const scalar_t* prefix_output, const float* prefix_lse, const scalar_t* suffix_output, const float* suffix_lse, float* output_lse, 
        const uint num_tokens, const uint num_heads, const uint head_size
    )
    {
        using pack_128b_t = uint4;
        const uint pack_size = 16/sizeof(scalar_t);
        const uint threads_per_head = head_size/pack_size;

        const uint globalId = blockIdx.x * NUM_THREADS + threadIdx.x;
        const uint tokens_heads_threads = num_tokens * num_heads * threads_per_head;

        if(globalId>=tokens_heads_threads) return;

        const uint token_head_idx = globalId / threads_per_head;
        const uint pack_idx = globalId % threads_per_head;

        const uint token_idx = token_head_idx / num_heads;
        const uint head_idx = token_head_idx % num_heads;

        const uint pack_offest = pack_idx * pack_size;
        const uint head_offest = token_idx * num_heads * head_size + head_idx * head_size;
        const scalar_t* prefix_head_ptr = prefix_output + head_offest;
        const scalar_t* suffix_head_ptr = suffix_output + head_offest;
        scalar_t * output_head_ptr = output + head_offest;

        float p_lse = prefix_lse[head_idx*num_tokens + token_idx];
        float s_lse = suffix_lse[head_idx*num_tokens + token_idx];
        p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
        s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

        float max_lse = fmax(p_lse,s_lse);
        p_lse = p_lse-max_lse;
        s_lse = s_lse-max_lse;
        const float p_se = expf(p_lse);
        const float s_se = expf(s_lse);
        const float out_se = p_se + s_se;
        const float p_scale = p_se/out_se;
        const float s_scale = s_se/out_se;

        if(pack_offest<head_size)
        {
            pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_idx];
            pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_idx];
            pack_128b_t o_output_pack;

            #pragma unroll
            for(int i = 0;i<pack_size;++i)
            {
                float p_out_f = vllm::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
                float s_out_f = vllm::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
                float o_out_f = p_out_f * p_scale + s_out_f * s_scale;
                vllm::from_float(reinterpret_cast<scalar_t*>(&o_output_pack)[i],o_out_f);
            }

            reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_idx] = o_output_pack;
        }

        if(output_lse != nullptr && pack_idx ==0)
        {
            float out_lse = log(out_se) + max_lse;
            output_lse[head_idx*num_tokens + token_idx] = out_lse;
        }

    }

}

#define DISPATCH_BY_DTYPE(scalar_dtype,fn)                              \
{                                                                       \
    if(scalar_dtype == at::ScalarType::Float){                          \
        fn(float);                                                      \
    }else if(scalar_dtype == at::ScalarType::Half){                     \
        fn(uint16_t);                                                   \
    }else if(scalar_dtype == at::ScalarType::BFloat16){                 \
        fn(__nv_bfloat16);                                              \
    }else{                                                              \
        TORCH_CHECK(false,"Unsupported data type of O: ", scalar_dtype); \
    }                                                                   \
}

#define LAUNCH_MERGE_ATTN_STATES(scalar_t,NUM_THREADS)                  \
{                                                                       \
    vllm::merge_attn_state_kernel<scalar_t,NUM_THREADS><<<grid,block>>>(\
        reinterpret_cast<scalar_t*>(output.data_ptr()),                 \
        reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),          \
        reinterpret_cast<float*>(prefix_lse.data_ptr()),             \
        reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),          \
        reinterpret_cast<float*>(suffix_lse.data_ptr()),             \
        output_lse_ptr,num_tokens,num_heads,head_size                       \
    );                                                                  \
}


template<typename scalar_t>
void merge_attn_state_launcher(torch::Tensor& output,
                                torch::Tensor& prefix_output,
                                torch::Tensor& prefix_lse,
                                torch::Tensor& suffix_output,
                                torch::Tensor& suffix_lse,
                                std::optional<torch::Tensor> output_lse)
{
    constexpr uint NUM_THREADS = 128;
    const uint num_tokens = output.size(0);
    const uint num_heads = output.size(1);
    const uint head_size = output.size(2);
    const uint pack_size = 16/sizeof(scalar_t);
    TORCH_CHECK(head_size%pack_size==0,"headsize must be multiple of pack_size:",pack_size);
    float* output_lse_ptr = nullptr;
    if(output_lse.has_value())
    {
        output_lse_ptr = output_lse.value().data_ptr<float>();
    }
    const uint threads_per_head = head_size/pack_size;
    const uint total_threads = num_tokens*num_heads*threads_per_head;

    dim3 block(NUM_THREADS);
    dim3 grid((total_threads+NUM_THREADS-1)/NUM_THREADS);

    LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                       \
{                                                                       \
    merge_attn_state_launcher<scalar_t>(output,prefix_output,prefix_lse,\
                                        suffix_output,suffix_lse,       \
                                        output_lse);                    \
}

void merge_attn_states(torch::Tensor& output,
                        torch::Tensor& prefix_output,
                        torch::Tensor& prefix_lse,
                        torch::Tensor& suffix_output,
                        torch::Tensor& suffix_lse,
                        std::optional<torch::Tensor> output_lse
)
{
    DISPATCH_BY_DTYPE(output.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
}