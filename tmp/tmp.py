import os
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer


BASE_DIR = "/home/chengzhang/RetrievalAttention2"
DATASETS = [
    "Qwen2.5-7B-128k-MultiKey3-0",
    "Qwen2.5-7B-128k-MultiKey3-8",
    "Qwen2.5-7B-128k-FWE-0",
    "Qwen2.5-7B-128k-FWE-8",
    "Qwen2.5-7B-128k-VT-0",
    "Qwen2.5-7B-128k-VT-8",
]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_data(dataset: str, layer_idx: int):
    layer_dir = f"{BASE_DIR}/data/{dataset}/layer_{layer_idx:02}"
    q = torch.load(f"{layer_dir}/q.pt")
    k = torch.load(f"{layer_dir}/k.pt")
    v = torch.load(f"{layer_dir}/v.pt")
    o = torch.load(f"{layer_dir}/o.pt")
    q_rope = torch.load(f"{layer_dir}/q_rope.pt")
    k_rope = torch.load(f"{layer_dir}/k_rope.pt")
    return q, k, v, o, q_rope, k_rope


def calc_rope(position_ids: torch.Tensor, head_dim: int, rope_base: float = 1000000.0):
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim))
    emb = position_ids[:, None] * inv_freq[None, :]
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    return cos, sin


def calc_eqk(q: torch.Tensor, k: torch.Tensor, head_idx: int, rope_base: float = 1000000.0):
    batch_size, num_q_heads, num_tokens, head_dim = q.shape
    position_ids = torch.arange(0, num_tokens, dtype=torch.float32, device=q.device)
    cos, sin = calc_rope(position_ids, head_dim=head_dim, rope_base=rope_base)
    num_k_heads = k.shape[1]
    group_size = num_q_heads // num_k_heads
    head_idx_k = head_idx // group_size
    half_head_dim = head_dim // 2
    k_rotate = torch.concat([k[0, head_idx_k, :, half_head_dim:], k[0, head_idx_k, :, :half_head_dim]], dim=-1)
    eq = torch.mean(q[0, head_idx], dim=0)
    ek = torch.mean(k[0, head_idx_k], dim=0)
    stdq = torch.std(q[0, head_idx], dim=0)
    stdk = torch.std(k[0, head_idx_k], dim=0)
    eqk0 = torch.mean(q[0, head_idx] * k[0, head_idx_k], dim=0)
    eqk1 = torch.mean(q[0, head_idx] * k_rotate, dim=0)
    # es = torch.sum(
    #     cos * (eq[None, :half_head_dim] * ek[None, :half_head_dim] + eq[None, half_head_dim:] * ek[None, half_head_dim:]) - \
    #     sin * (eq[None, half_head_dim:] * ek[None, :half_head_dim] - eq[None, :half_head_dim] * ek[None, half_head_dim:]),
    #     dim=-1,
    # )
    es = torch.sum(
        cos * (eqk0[None, :half_head_dim] + eqk0[None, half_head_dim:]) - \
        sin * (eqk1[None, half_head_dim:] - eqk1[None, :half_head_dim]),
        dim=-1,
    )
    return eq, ek, stdq, stdk, eqk0, eqk1, es


def get_e_map(
    es: torch.Tensor,
    eq: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    head_dim: int,
    p: float = 6.0,
    max_tokens: int = -1,
    stride: int = 32,
    num_samples: int = 128,
    block_size: int = 128,
):
    num_tokens = es.shape[0]
    if max_tokens > 0:
        num_tokens = max_tokens
        es = es[:, :, :num_tokens, :]
    arange = torch.arange(0, num_tokens, stride, device=es.device)
    # e_map = arange[:, None] >= arange[None, :]
    # e_map = torch.zeros((arange.shape[0], arange.shape[0]), dtype=torch.float32, device=es.device)

    threshold_0 = torch.max(es).item() - p * (head_dim ** 0.5)
    threshold_1 = torch.max(torch.sum(eq[None, :] * k, dim=-1, dtype=torch.float32)).item() - p * (head_dim ** 0.5)
    threshold = max(threshold_0, threshold_1)
    sample_stride = (num_tokens + num_samples - 1) // num_samples
    qk = torch.einsum('mk, nk -> mn', q_rope[::sample_stride], k_rope)
    mask = torch.arange(0, num_tokens, sample_stride, device=k.device)[:, None] >= torch.arange(num_tokens, device=k.device)[None, :]
    qk = torch.where(mask, qk, float('-inf'))
    max_qk = qk.max(dim=-1).values
    threshold_2 = max_qk.median().item() - p * (head_dim ** 0.5)
    # threshold = threshold_2

    # cnt = 0
    # for i in range(0, num_tokens, stride):
    #     local_cnt = 0
    #     local_width = min(stride, num_tokens - i)
    #     for ii in range(local_width):
    #         if es[i + ii] > threshold:
    #             cnt += num_tokens - (i + ii)
    #             local_cnt += 1
    #             # break
    #     if local_cnt > 0:
    #         # e_map = torch.where(arange[:, None] - arange[None, :] == i, 1.0, e_map)
    #         e_map = torch.where(arange[:, None] - arange[None, :] == i, local_cnt / local_width, e_map)

    q_rope = torch.nn.functional.pad(q_rope, (0, 0, 0, block_size - 1 - (num_tokens - 1) % block_size))
    k_rope = torch.nn.functional.pad(k_rope, (0, 0, 0, stride - 1 - (num_tokens - 1) % stride))
    q_avg = q_rope.reshape(-1, block_size, head_dim).mean(dim=1)  # [M, D]
    qk = torch.einsum('mk, nk -> mn', q_avg, k_rope)  # [M, N]
    block_arange = torch.arange(0, num_tokens, block_size, device=es.device)
    full_arange = torch.arange(0, k_rope.shape[0], device=es.device)
    qk.masked_fill_(block_arange[:, None] < full_arange[None, :], float('-inf'))
    qk_max = qk.max(dim=-1).values
    e_map = torch.where(qk > qk_max[:, None] - p * (head_dim ** 0.5), 1.0, 0.0)
    e_map = e_map.reshape(e_map.shape[0], -1, stride).mean(dim=-1)
    e_map = torch.tile(e_map.reshape(-1, 1, e_map.shape[-1]), (1, block_size // stride, 1)).reshape(-1, e_map.shape[-1])
    cnt = e_map.sum(dtype=torch.float64).item() * stride * stride

    sparsity = 1.0 - cnt / (num_tokens * (num_tokens + 1) // 2)
    return e_map, threshold_0, threshold_1, threshold_2, sparsity


def get_attn_map(
    q: torch.Tensor,  # [B, Hq, N, D],
    k: torch.Tensor,  # [B, Hk, N, D],
    head_idx: int,
    off_q: int,
    off_k: int,
    max_tokens: int = -1,
    stride: int = 32,
    top_p: float = 0.85,
):
    batch_size, num_q_heads, num_tokens, head_dim = q.shape
    num_k_heads = k.shape[1]
    group_size = num_q_heads // num_k_heads
    head_idx_k = head_idx // group_size
    if max_tokens > 0:
        num_tokens = max_tokens
        q = q[:, :, :num_tokens, :]
        k = k[:, :, :num_tokens, :]
    # assert off_q >= off_k
    qk = torch.einsum('mk, nk -> mn', q[0, head_idx, off_q::stride], k[0, head_idx_k])
    qk = qk.to(torch.float32) / np.sqrt(head_dim)
    arange = torch.arange(num_tokens, device=q.device)
    causal_mask = arange[off_q::stride, None] >= arange[None, :]
    qk = torch.where(causal_mask, qk, float('-inf'))
    w = torch.softmax(qk, dim=-1)
    sorted_w, sorted_i = torch.sort(w, dim=-1, descending=True)
    topp_mask = torch.cumsum(sorted_w, dim=-1) < top_p
    attn_map = torch.scatter(torch.zeros_like(w), -1, sorted_i, topp_mask.to(torch.float32))
    attn_map = torch.where(causal_mask, attn_map, 0.0)
    sparsity = 1.0 - attn_map.sum(dtype=torch.float64).item() / (attn_map.numel() / 2)
    attn_map = torch.nn.functional.pad(attn_map, (0, stride - 1 - (num_tokens - 1) % stride, 0, 0))
    # attn_map = torch.max(attn_map.reshape(attn_map.shape[0], -1, stride), dim=-1).values
    attn_map = torch.mean(attn_map.reshape(attn_map.shape[0], -1, stride), dim=-1)
    # attn_map = attn_map[:, off_k::stride]
    return attn_map, sparsity


def plot_attn_map(
    dataset: str,
    layer_idx: int,
    head_idx: int,
    p: float = 12.0,
    stride: int = 64,
    output_dir: str = None,
):
    q, k, v, o, q_rope, k_rope = get_data(dataset=dataset, layer_idx=layer_idx)
    batch_size, num_q_heads, num_tokens, head_dim = q.shape
    num_k_heads = k.shape[1]
    half_head_dim = head_dim // 2
    group_size = num_q_heads // num_k_heads
    head_idx_k = head_idx // group_size

    attn_map, real_sparsity = get_attn_map(q_rope, k_rope, head_idx=head_idx, off_q=0, off_k=0, max_tokens=-1, stride=stride)
    eq, ek, stdq, stdk, eqk0, eqk1, es = calc_eqk(q, k, head_idx=head_idx)
    e_map, thres_0, thres_1, thres_2, estimated_sparsity = get_e_map(
        es, eq, k[0, head_idx_k], q_rope[0, head_idx], k_rope[0, head_idx_k],
        # es, q_rope[0, head_idx].mean(dim=0), k_rope[0, head_idx_k], q_rope[0, head_idx], k_rope[0, head_idx_k],
        p=p, head_dim=head_dim, max_tokens=-1, stride=stride, num_samples=256,
    )

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(48, 32))

    sns.heatmap(attn_map.cpu().numpy(), ax=ax0, vmin=0.0, vmax=1.0, cbar=False)
    ax0.set_title(f'Real Attention Score Map ({real_sparsity * 100:3.2f}%)')

    sns.heatmap(e_map.cpu().numpy(), ax=ax1, vmin=0.0, vmax=1.0, cbar=False)
    ax1.set_title(f'Estimated Attention Score Map ({estimated_sparsity * 100:3.2f}%)')

    ax2.plot(es.cpu().numpy(), label='$E[P_{i-j}]$')
    ax2.plot([0, num_tokens - 1], [thres_0, thres_0], label='$\max_{i-j}(E[P_{i-j}])$')
    ax2.plot([0, num_tokens - 1], [thres_1, thres_1], label='$\max_j(E[Q]K_j)$')
    # ax2.plot([0, num_tokens - 1], [thres_2, thres_2], label='$med_i(\max_j(Q_iR_{i-j}K_j))$')
    ax2.set_xlabel('Relative Position $i-j$')
    ax2.set_title('$E[P_{i-j}]$')
    ax2.legend()

    ax3.plot(eq[:half_head_dim].to(torch.float32).cpu().numpy(), label='$E[Q_{dI}]$', color='C0', ls='--')
    ax3.plot(eq[half_head_dim:].to(torch.float32).cpu().numpy(), label='$E[Q_{dR}]$', color='C0', ls='-')
    ax3.plot(ek[:half_head_dim].to(torch.float32).cpu().numpy(), label='$E[K_{dI}]$', color='C1', ls='--')
    ax3.plot(ek[half_head_dim:].to(torch.float32).cpu().numpy(), label='$E[K_{dR}]$', color='C1', ls='-')
    ax3.set_xlabel('Channel $d$')
    ax3.set_title('$E[Q]$ and $E[K]$')
    ax3.legend()

    ax4.plot(stdq[:half_head_dim].to(torch.float32).cpu().numpy(), label='$Var[Q_{dI}]$', color='C0', ls='--')
    ax4.plot(stdq[half_head_dim:].to(torch.float32).cpu().numpy(), label='$Var[Q_{dR}]$', color='C0', ls='-')
    ax4.plot(stdk[:half_head_dim].to(torch.float32).cpu().numpy(), label='$Var[K_{dI}]$', color='C1', ls='--')
    ax4.plot(stdk[half_head_dim:].to(torch.float32).cpu().numpy(), label='$Var[K_{dR}]$', color='C1', ls='-')
    ax4.set_xlabel('Channel $d$')
    ax4.set_title('$Var[Q]$ and $Var[K]$')
    ax4.set_ylim([0, max(eq.abs().max().item(), ek.abs().max().item())])
    ax4.legend()

    ax5.plot(eqk0[:half_head_dim].to(torch.float32).cpu().numpy(), label='$E[Q_{dI} \\times K_{dI}]$', color='C0', ls='--')
    ax5.plot(eqk0[half_head_dim:].to(torch.float32).cpu().numpy(), label='$E[Q_{dR} \\times K_{dR}]$', color='C0', ls='-')
    ax5.plot(eqk1[:half_head_dim].to(torch.float32).cpu().numpy(), label='$E[Q_{dI} \\times K_{dR}]$', color='C1', ls='--')
    ax5.plot(eqk1[half_head_dim:].to(torch.float32).cpu().numpy(), label='$E[Q_{dR} \\times K_{dI}]$', color='C1', ls='-')
    ax5.set_xlabel('Channel $d$')
    ax5.set_title('$E[QK]$')
    ax5.legend()

    plt.suptitle(f'Layer #{layer_idx:02}; Head #{head_idx:02}')
    if output_dir is not None:
        plt.savefig(f'{output_dir}/layer_{layer_idx:02}_head_{head_idx:02}.png')
        plt.close(fig)
    return real_sparsity, estimated_sparsity


if __name__ == '__main__':
    for dataset in DATASETS:
        print('=' * 80)
        print(f'Dataset: {dataset}')
        log_path = f"{BASE_DIR}/logs/{dataset}.txt"
        output_dir = f"{BASE_DIR}/img/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write('')
        rs_list = []
        es_list = []
        for layer_idx in range(0, 28, 2):
            for head_idx in range(0, 28, 2):
                # print(f'Layer #{layer_idx:02}; Head #{head_idx:02}')
                # plot_attn_map(layer_idx, head_idx, save=True)
                rs, es = plot_attn_map(dataset, layer_idx, head_idx, output_dir=output_dir)
                print(f'Layer #{layer_idx:02}; Head #{head_idx:02}: Sparsity = {rs * 100:3.2f}% -> {es * 100:3.2f}%')
                with open(log_path, 'a') as f:
                    f.write(f'Layer #{layer_idx:02}; Head #{head_idx:02}: Sparsity = {rs * 100:3.2f}% -> {es * 100:3.2f}%\n')
                rs_list.append(rs)
                es_list.append(es)
                torch.cuda.empty_cache()
        print(f'Overall Sparsity: {sum(rs_list) / len(rs_list) * 100:3.2f}% -> {sum(es_list) / len(es_list) * 100:3.2f}%')
        with open(log_path, 'a') as f:
            f.write(f'Overall Sparsity: {sum(rs_list) / len(rs_list) * 100:3.2f}% -> {sum(es_list) / len(es_list) * 100:3.2f}%\n')
        print()
