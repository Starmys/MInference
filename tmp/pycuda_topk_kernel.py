import torch
import numpy as np
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule



_partially_apply_rope_kernel_code = '''
#include <queue>
#include <cuda_bf16.h>

extern "C" {

__global__ void PYCUDA_ROPE_KERNEL(
    short* x, // [B, N, H, D]
    float* f,  // [1, N, D]
    long long* idx,  // [N]
    short* y,  // [B, M, H, D]
    const int M,
    const int N,
    const int H,
    const int D
) {
    const int n_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (n_idx >= N) return;
    int m_idx = idx[n_idx];
    x += blockIdx.y * N * H * D + n_idx * H * D + threadIdx.x * D;
    y += blockIdx.y * M * H * D + m_idx * H * D + threadIdx.x * D;
    f += n_idx * D;
    
    std::priority_queue<int> max_priority_queue;

    float4 buf_a;
    float4 buf_b;
    float4 buf_res;

    float2 complex_a;
    float2 complex_b;
    __nv_bfloat162 complex_res;

    // TODO: unroll
    for (int offset = 0; offset < D; offset += 8) {
        # pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i % 4 == 0) buf_a = (reinterpret_cast<float4*>(&x[offset]))[0];
            if (i % 2 == 0) buf_b = (reinterpret_cast<float4*>(&f[offset + i * 2]))[0];
            complex_a = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&buf_a))[i]);
            complex_b = (reinterpret_cast<float2*>(&buf_b))[i % 2];
            complex_res = __float22bfloat162_rn(complex_multiplication(complex_a, complex_b));
            (reinterpret_cast<float*>(&buf_res))[i] = (reinterpret_cast<float*>(&complex_res))[0];
        }
        (reinterpret_cast<float4*>(&y[offset]))[0] = buf_res;
    }
}

}
'''
_partially_apply_rope_kernel = SourceModule(
    _partially_apply_rope_kernel_code,
    options=['-std=c++14', '-O3'],
    no_extern_c=True,
).get_function(f'PYCUDA_ROPE_KERNEL')


def _partially_apply_rope(
    x: torch.Tensor,  # [B, N, H, D], torch.bfloat16
    f: torch.Tensor,  # [1, N, D // 2], torch.complex64
    idx: torch.Tensor,  # [N], torch.int32
    y: torch.Tensor,  # [B, M, H, D], torch.bfloat16
):
    B, N, H, D = x.shape
    M = y.shape[1]
    thead_num = 128
    block_N = thead_num // H
    _partially_apply_rope_kernel(
        x.view(torch.int16),
        torch.view_as_real(f),
        idx,
        y.view(torch.int16),
        np.int32(M), np.int32(N), np.int32(H), np.int32(D),
        grid=(triton.cdiv(N, block_N), B, 1),
        block=(H, block_N, 1),
    )
