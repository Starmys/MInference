import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_kernel(
    x_ptr, mask_ptr,
    num_valid_centroids, sm_scale,
    stride_xz, stride_xh, stride_xn,
    stride_mz, stride_mn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < num_valid_centroids

    mask_ptrs = mask_ptr + pid_z * stride_mz + offs_n * stride_mn
    x_ptrs = x_ptr + pid_z * stride_xz + pid_h * stride_xh + offs_n * stride_xn
    x = tl.load(x_ptrs, mask=mask_n, other=float('-inf'))

    x = x.to(tl.float32) * sm_scale
    m = tl.max(x, axis=0, keep_dims=True)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0, keep_dims=True)
    y = e / s

    mask = tl.load(mask_ptrs, mask=mask_n, other=0)
    y = tl.where(mask, 0., y)

    tl.store(x_ptrs, y.to(x_ptr.type.element_ty), mask=mask_n)


def fused_gemm_softmax(
    query: torch.Tensor,                # [batch_size, num_q_tokens, num_q_heads, head_dim]
    centroids: torch.Tensor,            # [batch_size, num_kv_heads, num_centroids, head_dim]
    mask: torch.Tensor,                 # [batch_size, num_kv_heads, num_centroids]
    num_valid_centroids: torch.Tensor,  # [1, ]
    qk_buffer: torch.Tensor,            # [batch_size, num_kv_heads, group_size, num_centroids]
    out: torch.Tensor,                  # [batch_size, num_kv_heads, num_centroids]
    sm_scale: float = None,
):
    batch_size, num_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, num_centroids, _ = centroids.shape

    assert num_q_tokens == 1
    group_size = num_q_heads // num_kv_heads
    sm_scale = sm_scale or head_dim ** -0.5

    num_valid_centroids = num_valid_centroids.item()
    query = query.view(-1, group_size, head_dim)
    centroids = centroids.view(-1, num_centroids, head_dim).swapaxes(1, 2)[..., :num_valid_centroids]
    qk_buffer = qk_buffer.view(-1, group_size, num_centroids)

    torch.bmm(query, centroids, out=qk_buffer[..., :num_valid_centroids])

    assert num_centroids & (num_centroids - 1) == 0
    _masked_softmax_kernel[(group_size, batch_size * num_kv_heads, )](
        qk_buffer, mask, num_valid_centroids, sm_scale,
        qk_buffer.stride(0), qk_buffer.stride(1), qk_buffer.stride(2),
        mask.stride(1), mask.stride(2),
        BLOCK_SIZE=num_centroids, num_warps=4,
    )

    torch.sum(qk_buffer.view(batch_size, num_kv_heads, group_size, num_centroids), dim=2, out=out)

    return out[..., :num_valid_centroids]


# @torch.compile
def ref_gemm_softmax(
    query: torch.Tensor,                # [batch_size, num_q_tokens, num_q_heads, head_dim]
    centroids: torch.Tensor,            # [batch_size, num_kv_heads, num_centroids, head_dim]
    mask: torch.Tensor,                 # [batch_size, num_kv_heads, num_centroids]
    num_valid_centroids: torch.Tensor,  # [1, ]
    sm_scale: float = None,
):
    batch_size, num_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, num_centroids, _ = centroids.shape

    assert num_q_tokens == 1
    group_size = num_q_heads // num_kv_heads
    sm_scale = sm_scale or head_dim ** -0.5
    query = query.view(batch_size, num_kv_heads, group_size, head_dim)
    key = centroids[:, :, :num_valid_centroids, :]
    mask = mask[:, :, :num_valid_centroids]

    p = torch.einsum('bhgd, bhnd -> bhgn', query, key) * sm_scale
    p = torch.softmax(p, dim=-1)
    s = torch.sum(p, dim=2).masked_fill_(mask, 0)

    return s


def profile(func, inputs, num_warmups=0, num_iters=1):
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(*inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / num_iters
    return latency


def main(
    batch_size: int = 1,
    num_q_tokens: int = 1,
    num_centroids: int = 8192,
    num_valid_centroids_val: int = 6789,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
):
    torch.manual_seed(seed)
    query = torch.randn(batch_size, num_q_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    centroids = torch.randn(batch_size, num_kv_heads, num_centroids, head_dim, dtype=dtype, device=device)
    mask = torch.randint(0, 2, (batch_size, num_kv_heads, num_centroids), dtype=torch.bool, device=device)
    num_valid_centroids = torch.tensor([num_valid_centroids_val], dtype=torch.int32, device=device)
    qk_buffer = torch.empty(batch_size, num_kv_heads, num_q_heads // num_kv_heads, num_centroids, dtype=dtype, device=device)
    out_buffer = torch.empty(batch_size, num_kv_heads, num_centroids, dtype=dtype, device=device)

    ref = ref_gemm_softmax(query, centroids, mask, num_valid_centroids.item())
    out = fused_gemm_softmax(query, centroids, mask, num_valid_centroids, qk_buffer, out_buffer)
    # torch.testing.assert_close(ref, out)

    # inputs = (query, centroids, mask, num_valid_centroids.item())
    # print(f"ref_gemm_softmax: {profile(ref_gemm_softmax, inputs):.2f} ms")
    # inputs = (query, centroids, mask, num_valid_centroids, qk_buffer, out_buffer)
    # print(f"fused_gemm_softmax: {profile(fused_gemm_softmax, inputs):.2f} ms")


if __name__ == "__main__":
    main()
