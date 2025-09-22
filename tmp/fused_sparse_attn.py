import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_with_kvcache


def get_chunk_size_heuristic(batch_size, num_blocks, target_grid_size=1024):
    # batch_size * (num_blocks / chunk_size) >= target_grid_size
    return max(1, triton.cdiv(batch_size * num_blocks, target_grid_size))


@triton.jit
def _triton_attn_fwd_inner(
    q, acc, l_i, m_i,
    k_ptrs, v_ptrs, i_ptrs, w_ptrs,
    stride_kn, stride_vn, stride_in, stride_wn,
    lo, hi,
    offs_m, offs_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    WEIGHTED: tl.constexpr, SPARSED: tl.constexpr, MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        if MASKED:
            mask = cols < hi
        if SPARSED:
            if MASKED:
                cols = tl.load(i_ptrs + cols * stride_in, mask=mask, other=0)
            else:
                cols = tl.load(i_ptrs + cols * stride_in)
        if MASKED:
            k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=mask[None, :], other=0.)
        else:
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        if WEIGHTED:
            if MASKED:
                w = tl.load(w_ptrs + cols * stride_wn, mask=mask, other=0.)
            else:
                w = tl.load(w_ptrs + cols * stride_wn)
            k *= w[None, :]
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASKED:
            qk = tl.where(mask[None, :], qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        if MASKED:
            v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=mask[:, None], other=0.)
        else:
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(q.type.element_ty), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    return acc, l_i, m_i


@triton.jit
def _fused_retrieval_attention_kernel(
    query, key, value, selected_indices, valid_length,
    steady_key, steady_value, steady_len,
    centroids, value_sum, cluster_size, selected_centroids, start_index, max_consider_cluster_num,
    out, lse, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_k1z, stride_k1h, stride_k1n, stride_k1d,
    stride_v1z, stride_v1h, stride_v1n, stride_v1d,
    stride_i1z, stride_i1h, stride_i1n,
    stride_k2z, stride_k2h, stride_k2n, stride_k2d,
    stride_v2z, stride_v2h, stride_v2n, stride_v2d,
    stride_k3z, stride_k3h, stride_k3n, stride_k3d,
    stride_v3z, stride_v3h, stride_v3n, stride_v3d,
    stride_wz, stride_wh, stride_wn,
    stride_i3z, stride_i3h, stride_i3n,
    stride_on, stride_oz, stride_oh, stride_om, stride_od,
    stride_ln, stride_lz, stride_lh, stride_lm,
    B, H, M, N1, N2, N3, C, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M

    q_ptrs = query + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    q = (tl.load(q_ptrs, ) * (sm_scale * 1.44269504)).to(query.type.element_ty)

    if pid_n < N1:
        k_ptrs = key + pid_z * stride_k1z + pid_h * stride_k1h + offs_d[:, None] * stride_k1d
        v_ptrs = value + pid_z * stride_v1z + pid_h * stride_v1h + offs_d[None, :] * stride_v1d
        i_ptrs = selected_indices + pid_z * stride_i1z + pid_h * stride_i1h
        valid_length_val = tl.load(valid_length + pid_z * H + pid_h)
        start_n = pid_n * C * BLOCK_N
        end_n = tl.minimum(start_n + C * BLOCK_N, valid_length_val)
        if start_n < end_n:
            mid_n = end_n // BLOCK_N * BLOCK_N
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, i_ptrs, None,
                stride_k1n, stride_v1n, stride_i1n, None,
                start_n, mid_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=False, SPARSED=True, MASKED=False,
            )
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, i_ptrs, None,
                stride_k1n, stride_v1n, stride_i1n, None,
                mid_n, end_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=False, SPARSED=True, MASKED=True,
            )
    elif pid_n < N1 + N2:
        k_ptrs = steady_key + pid_z * stride_k2z + pid_h * stride_k2h + offs_d[:, None] * stride_k2d
        v_ptrs = steady_value + pid_z * stride_v2z + pid_h * stride_v2h + offs_d[None, :] * stride_v2d
        start_n = (pid_n - N1) * C * BLOCK_N
        end_n = tl.minimum(start_n + C * BLOCK_N, steady_len)
        if start_n < end_n:
            mid_n = end_n // BLOCK_N * BLOCK_N
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, None, None,
                stride_k2n, stride_v2n, None, None,
                start_n, mid_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=False, SPARSED=False, MASKED=False,
            )
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, None, None,
                stride_k2n, stride_v2n, None, None,
                mid_n, end_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=False, SPARSED=False, MASKED=True,
            )
    else:
        k_ptrs = centroids + pid_z * stride_k3z + pid_h * stride_k3h + offs_d[:, None] * stride_k3d
        v_ptrs = value_sum + pid_z * stride_v3z + pid_h * stride_v3h + offs_d[None, :] * stride_v3d
        i_ptrs = selected_centroids + pid_z * stride_i3z + pid_h * stride_i3h
        w_ptrs = cluster_size + pid_z * stride_wz + pid_h * stride_wh
        start_n = start_index + (pid_n - N1 - N2) * C * BLOCK_N
        end_n = tl.minimum(start_n + C * BLOCK_N, max_consider_cluster_num)
        if start_n < end_n:
            mid_n = start_index + (end_n - start_index) // BLOCK_N * BLOCK_N
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, i_ptrs, w_ptrs,
                stride_k3n, stride_v3n, stride_i3n, stride_wn,
                start_n, mid_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=True, SPARSED=True, MASKED=False,
            )
            acc, l_i, m_i = _triton_attn_fwd_inner(
                q, acc, l_i, m_i,
                k_ptrs, v_ptrs, i_ptrs, w_ptrs,
                stride_k3n, stride_v3n, stride_i3n, stride_wn,
                mid_n, end_n,
                offs_m, offs_n,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                WEIGHTED=True, SPARSED=True, MASKED=True,
            )

    l_ptrs = lse + pid_z * stride_lz + pid_h * stride_lh + pid_n * stride_ln + offs_m * stride_lm
    m_i = tl.math.log2(l_i) + m_i
    tl.store(l_ptrs, m_i.to(lse.type.element_ty), mask=mask_m)
    o_ptrs = out + pid_z * stride_oz + pid_h * stride_oh + pid_n * stride_on + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    acc /= (l_i[:, None] + 1e-6)
    tl.store(o_ptrs, acc.to(out.type.element_ty), mask=mask_m[:, None])


@triton.jit
def _combine_kernel(
    out_buffer, lse_buffer, out,
    stride_obc, stride_obz, stride_obh, stride_obd,
    stride_lbc, stride_lbz, stride_lbh,
    stride_oz, stride_oh, stride_od,
    num_chunks,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_h = tl.program_id(0)

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    mask_c = offs_c < num_chunks

    lb_ptrs = lse_buffer + off_z * stride_lbz + off_h * stride_lbh + offs_c * stride_lbc
    l = tl.load(lb_ptrs, mask=mask_c, other=float("-inf"))
    l_max = tl.max(l, axis=0, keep_dims=True)
    l_sum = tl.sum(tl.exp2(l - l_max), axis=0, keep_dims=True)
    l_all = tl.log2(l_sum) + l_max

    ob_ptrs = out_buffer + off_z * stride_obz + off_h * stride_obh + offs_c[:, None] * stride_obc + offs_d[None, :] * stride_obd
    o = tl.load(ob_ptrs, mask=mask_c[:, None], other=0.) * tl.exp2(l - l_all)[:, None]
    o = tl.sum(o, axis=0)

    o_ptrs = out + off_z * stride_oz + off_h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, o.to(out.type.element_ty))


def fused_retrieval_attention(
    # Retrieval
    query: torch.Tensor,               # [batch_size, num_q_tokens, num_q_heads, head_dim], bf16
    key: torch.Tensor,                 # [batch_size, num_kv_heads, num_k_tokens, head_dim], bf16
    value: torch.Tensor,               # [batch_size, num_kv_heads, num_k_tokens, head_dim], bf16
    selected_indices: torch.Tensor,    # [batch_size, num_kv_heads, buffer_size], int32
    valid_length: torch.Tensor,        # [batch_size, num_kv_heads], int32
    # Streaming
    steady_key: torch.Tensor,          # [batch_size, num_kv_heads, max_steady_size, head_dim], bf16
    steady_value: torch.Tensor,        # [batch_size, num_kv_heads, max_steady_size, head_dim], bf16
    steady_len: torch.Tensor,          # [1, ], int32
    # Centroids
    centroids: torch.Tensor,           # [batch_size, num_kv_heads, num_centroids, head_dim], bf16
    value_sum: torch.Tensor,           # [batch_size, num_kv_heads, num_centroids, head_dim], bf16
    cluster_size: torch.Tensor,        # [batch_size, num_kv_heads, num_centroids], bf16
    selected_centroids: torch.Tensor,  # [batch_size, num_kv_heads, max_consider_cluster_num], int64
    start_index: torch.Tensor,         # [1, ], int32
):
    batch_size, num_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, num_k_tokens, _ = key.shape
    buffer_size = selected_indices.shape[-1]
    max_steady_size = steady_key.shape[-2]
    num_centroids = centroids.shape[-2]
    max_consider_cluster_num = selected_centroids.shape[-1]
    steady_len = steady_len.item()
    start_index = start_index.item()
    sm_scale = head_dim ** -0.5

    assert num_q_tokens == 1
    group_size = num_q_heads // num_kv_heads
    query = query.view(batch_size, num_kv_heads, group_size, head_dim)

    block_M = 16
    block_N = 128
    block_D = head_dim
    num_blocks_1 = triton.cdiv(buffer_size, block_N)
    num_blocks_2 = triton.cdiv(steady_len, block_N)
    num_blocks_3 = triton.cdiv(max_consider_cluster_num - start_index, block_N)

    chunk_size = get_chunk_size_heuristic(batch_size, num_blocks_1 + num_blocks_2 + num_blocks_3)
    num_chunks_1 = triton.cdiv(num_blocks_1, chunk_size)
    num_chunks_2 = triton.cdiv(num_blocks_2, chunk_size)
    num_chunks_3 = triton.cdiv(num_blocks_3, chunk_size)
    num_chunks = num_chunks_1 + num_chunks_2 + num_chunks_3

    out_buffer = torch.empty(
        (num_chunks, batch_size, num_kv_heads, group_size, head_dim), 
        dtype=torch.float32, device=query.device
    )
    lse_buffer = torch.empty(
        (num_chunks, batch_size, num_kv_heads, group_size), 
        dtype=torch.float32, device=query.device
    )

    _fused_retrieval_attention_kernel[(num_chunks, num_kv_heads, batch_size)](
        # Pointers
        query, key, value, selected_indices, valid_length,
        steady_key, steady_value, steady_len,
        centroids, value_sum, cluster_size, selected_centroids, start_index, max_consider_cluster_num,
        out_buffer, lse_buffer, sm_scale,
        # Strides: part 1
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        selected_indices.stride(0), selected_indices.stride(1), selected_indices.stride(2),
        # Strides: part 2
        steady_key.stride(0), steady_key.stride(1), steady_key.stride(2), steady_key.stride(3),
        steady_value.stride(0), steady_value.stride(1), steady_value.stride(2), steady_value.stride(3),
        # Strides: part 3
        centroids.stride(0), centroids.stride(1), centroids.stride(2), centroids.stride(3),
        value_sum.stride(0), value_sum.stride(1), value_sum.stride(2), value_sum.stride(3),
        cluster_size.stride(0), cluster_size.stride(1), cluster_size.stride(2),
        selected_centroids.stride(0), selected_centroids.stride(1), selected_centroids.stride(2),
        # Strides: output
        out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3), out_buffer.stride(4),
        lse_buffer.stride(0), lse_buffer.stride(1), lse_buffer.stride(2), lse_buffer.stride(3),
        # Dimensions: B, H, M, N1, N2, N3, C, D
        batch_size, num_kv_heads, group_size, num_chunks_1, num_chunks_2, num_chunks_3, chunk_size, head_dim,
        # Block sizes
        BLOCK_M=block_M, BLOCK_N=block_N, BLOCK_D=block_D,
        num_warps=4, num_stages=2,
    )

    out_buffer = out_buffer.view(num_chunks, batch_size, num_q_heads, head_dim)
    lse_buffer = lse_buffer.view(num_chunks, batch_size, num_q_heads)
    out = torch.empty((batch_size, num_q_tokens, num_q_heads, head_dim), dtype=query.dtype, device=query.device)

    _combine_kernel[(num_q_heads, batch_size)](
        out_buffer, lse_buffer, out,
        out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3),
        lse_buffer.stride(0), lse_buffer.stride(1), lse_buffer.stride(2),
        out.stride(0), out.stride(2), out.stride(3),
        num_chunks, BLOCK_C=triton.next_power_of_2(num_chunks), BLOCK_D=head_dim,
    )

    return out


def ref_retrieval_attention(
    # Retrieval
    query: torch.Tensor,               # [batch_size, num_q_tokens, num_q_heads, head_dim], bf16
    key: torch.Tensor,                 # [batch_size, num_kv_heads, num_k_tokens, head_dim], bf16
    value: torch.Tensor,               # [batch_size, num_kv_heads, num_k_tokens, head_dim], bf16
    selected_indices: torch.Tensor,    # [batch_size, num_kv_heads, buffer_size], int32
    valid_length: torch.Tensor,        # [batch_size, num_kv_heads], int32
    # Streaming
    steady_key: torch.Tensor,          # [batch_size, num_kv_heads, max_steady_size, head_dim], bf16
    steady_value: torch.Tensor,        # [batch_size, num_kv_heads, max_steady_size, head_dim], bf16
    steady_len: torch.Tensor,          # [1, ], int32
    # Centroids
    centroids: torch.Tensor,           # [batch_size, num_kv_heads, num_centroids, head_dim], bf16
    value_sum: torch.Tensor,           # [batch_size, num_kv_heads, num_centroids, head_dim], bf16
    cluster_size: torch.Tensor,        # [batch_size, num_kv_heads, num_centroids], bf16
    selected_centroids: torch.Tensor,  # [batch_size, num_kv_heads, max_consider_cluster_num], int64
    start_index: torch.Tensor,         # [1, ], int32
):
    batch_size, num_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, num_k_tokens, _ = key.shape
    buffer_size = selected_indices.shape[-1]
    max_steady_size = steady_key.shape[-2]
    num_centroids = centroids.shape[-2]
    max_consider_cluster_num = selected_centroids.shape[-1]
    steady_len = steady_len.item()
    start_index = start_index.item()
    sm_scale = head_dim ** -0.5

    assert num_q_tokens == 1
    group_size = num_q_heads // num_kv_heads
    query = query.view(batch_size, num_kv_heads, group_size, head_dim)

    out = torch.empty_like(query)
    for b in range(batch_size):
        for h in range(num_kv_heads):
            q = query[b, h].to(torch.float32) * sm_scale
            k1 = key[b, h, selected_indices[b, h, :valid_length[b, h]], :].to(torch.float32)
            v1 = value[b, h, selected_indices[b, h, :valid_length[b, h]], :].to(torch.float32)
            qk1 = torch.matmul(q, k1.T)
            k2 = steady_key[b, h, :steady_len, :].to(torch.float32)
            v2 = steady_value[b, h, :steady_len, :].to(torch.float32)
            qk2 = torch.matmul(q, k2.T)
            k3 = centroids[b, h, selected_centroids[b, h, start_index:], :].to(torch.float32)
            v3 = value_sum[b, h, selected_centroids[b, h, start_index:], :].to(torch.float32)
            w3 = cluster_size[b, h, selected_centroids[b, h, start_index:]].to(torch.float32)
            qk3 = torch.matmul(q, k3.T) * w3[None, :]
            qk = torch.cat([qk1, qk2, qk3], dim=1)
            qk = torch.softmax(qk, dim=-1)
            v = torch.cat([v1, v2, v3], dim=0)
            out[b, h] = torch.matmul(qk, v)

    out = out.view(batch_size, num_q_tokens, num_q_heads, head_dim).to(query.dtype)
    return out


def profile(func, inputs, num_warmups=50, num_iters=50):
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
    batch_size: int = 3,
    num_q_tokens: int = 1,
    num_k_tokens: int = 131072,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    buffer_size: int = 8192,
    max_steady_size: int = 4096,
    steady_len_val: int = 3579,
    # steady_len_val: int = 0,
    num_centroids: int = 8192,
    max_consider_cluster_num: int = 4096,
    start_index_val: int = 1234,
    # start_index_val: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
):
    torch.manual_seed(seed)
    query = torch.randn(batch_size, num_q_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch_size, num_kv_heads, num_k_tokens, head_dim, dtype=dtype, device=device)
    value = torch.rand(batch_size, num_kv_heads, num_k_tokens, head_dim, dtype=dtype, device=device)
    selected_indices = torch.randint(0, num_k_tokens, (batch_size, num_kv_heads, buffer_size), dtype=torch.int32, device=device)
    valid_length = torch.randint(buffer_size // 2, buffer_size, (batch_size, num_kv_heads), dtype=torch.int32, device=device)
    # valid_length = torch.zeros(batch_size, num_kv_heads, dtype=torch.int32, device=device) + buffer_size
    steady_key = torch.randn(batch_size, num_kv_heads, max_steady_size, head_dim, dtype=dtype, device=device)
    steady_value = torch.randn(batch_size, num_kv_heads, max_steady_size, head_dim, dtype=dtype, device=device)
    steady_len = torch.tensor([steady_len_val], dtype=torch.int32, device=device)
    centroids = torch.randn(batch_size, num_kv_heads, num_centroids, head_dim, dtype=dtype, device=device)
    value_sum = torch.randn(batch_size, num_kv_heads, num_centroids, head_dim, dtype=dtype, device=device)
    cluster_size = torch.rand(batch_size, num_kv_heads, num_centroids, dtype=dtype, device=device)
    selected_centroids = torch.randint(0, num_centroids, (batch_size, num_kv_heads, max_consider_cluster_num), dtype=torch.int64, device=device)
    start_index = torch.tensor([start_index_val], dtype=torch.int32, device=device)

    ref = ref_retrieval_attention(
        query, key, value, selected_indices, valid_length,
        steady_key, steady_value, steady_len,
        centroids, value_sum, cluster_size, selected_centroids, start_index
    )
    out = fused_retrieval_attention(
        query, key, value, selected_indices, valid_length,
        steady_key, steady_value, steady_len,
        centroids, value_sum, cluster_size, selected_centroids, start_index
    )
    torch.testing.assert_close(ref, out)
    # Greatest absolute difference: 0.001953125 at index (0, 0, 2, 61) (up to 1e-10 allowed)
    # Greatest relative difference: 0.00775146484375 at index (1, 0, 10, 102) (up to 1e-10 allowed)
    # import ipdb; ipdb.set_trace()

    concat_k = torch.concat([key[:, :, :buffer_size], steady_key[:, :, :steady_len], centroids[:, :, start_index:]], dim=2).swapaxes(1, 2)
    concat_v = torch.concat([value[:, :, :buffer_size], steady_value[:, :, :steady_len], value_sum[:, :, start_index:]], dim=2).swapaxes(1, 2)
    print(concat_k.shape, concat_v.shape)

    inputs = (query, key, value, selected_indices, valid_length,
              steady_key, steady_value, steady_len,
              centroids, value_sum, cluster_size, selected_centroids, start_index)
    latency = profile(fused_retrieval_attention, inputs)
    print(f"fused_retrieval_attention: {latency:.2f} ms")

    inputs = (query, concat_k, concat_v)
    latency = profile(flash_attn_with_kvcache, inputs)
    print(f"flash_attn_with_kvcache: {latency:.2f} ms")


if __name__ == "__main__":
    main()
