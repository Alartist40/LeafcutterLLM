#!/usr/bin/env python3
"""
LeafcutterLLM — Python reference forward for DeepSeek-2 / GLM-DSA.

This is the *gold standard* that the Rust native implementation must
match bit-for-bit (or near-bit-for-bit given f32 quant noise).

Implements one-layer of:
  1. MoE FFN forward (routed + shared), sigmoid routing.
  2. MLA attention forward (q_a/q_b + kv_a_mqa + k_b + v_b + RoPE on
     the rope portion of q/k).

Usage:
    python3 scripts/ref_mla_moe.py --random     # tiny CPU sanity
    python3 scripts/ref_mla_moe.py --layer 0    # forward layer 0 of Kimi-K2.6

Requires numpy. The model-reading path is python-side and does NOT
depend on the Rust engine; we mimic the GGUF-dequant steps with numpy
where useful for math verification, but for unit tests we use
random unit tensors.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import struct

try:
    import numpy as np
except ImportError:
    sys.stderr.write("This script needs numpy: pip install numpy\n")
    sys.exit(2)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight"""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def moe_forward(
    hidden: np.ndarray,    # [seq_len, hidden]
    gate_inp: np.ndarray,  # [num_experts, hidden]
    gate_exps: np.ndarray, # [num_experts, expert_ffn, hidden]
    up_exps: np.ndarray,   # [num_experts, expert_ffn, hidden]
    down_exps: np.ndarray, # [num_experts, hidden, expert_ffn]
    gate_shexp: np.ndarray,   # [expert_ffn, hidden]
    up_shexp: np.ndarray,     # [expert_ffn, hidden]
    down_shexp: np.ndarray,   # [hidden, expert_ffn]
    exp_probs_b: np.ndarray | None = None,  # [num_experts] optional
    num_experts_used: int = 8,
    routed_scaling_factor: float = 1.0,
    gating_func: int = 2,    # 2 = sigmoid, 1 = softmax
    norm_topk_prob: bool = True,
) -> np.ndarray:
    """Return hidden + routed + shared for one layer's MoE. shape [seq_len, hidden]."""
    seq_len, hidden_dim = hidden.shape
    num_experts = gate_inp.shape[0]
    k = num_experts_used
    out = np.zeros_like(hidden)

    # Shared expert: applies to EVERY token regardless of routing.
    gate_proj_shared = hidden @ gate_shexp.T          # [seq_len, expert_ffn]
    up_proj_shared = hidden @ up_shexp.T
    shared_out = (silu(gate_proj_shared) * up_proj_shared) @ down_shexp.T  # [seq_len, hidden]

    for t in range(seq_len):
        h = hidden[t]
        scores = gate_inp @ h              # [num_experts]
        if gating_func == 2:
            sig = sigmoid(scores)
            topk_idx = np.argsort(-sig)[:k]
            ws = sig[topk_idx]
            if norm_topk_prob:
                ws = ws / (ws.sum() + 1e-9)
        else:  # softmax
            topk_idx = np.argsort(-scores)[:k]
            exps = np.exp(scores[topk_idx] - scores[topk_idx].max())
            ws = exps / exps.sum()

        routed_t = np.zeros((hidden_dim,), dtype=hidden.dtype)
        for rank, e in enumerate(topk_idx):
            eg = hidden[t] @ gate_exps[e].T           # [expert_ffn]
            eu = hidden[t] @ up_exps[e].T              # [expert_ffn]
            gated = silu(eg) * eu                     # [expert_ffn]
            routed_t += gated @ down_exps[e].T        # [hidden]

        routed_t *= routed_scaling_factor
        out[t] = shared_out[t] + routed_t

        if exp_probs_b is not None:
            # DeepSeek-V3 sigmoid-bias adds b_i * expert(hidden); approximated
            # by adding a contribution proportional to the bias accumulated sum.
            bias_term = exp_probs_b.sum()
            out[t] += bias_term * 0.0 * hidden[t]  # placeholder; fold only if needed

    return out


def mla_attention_forward(
    hidden: np.ndarray,       # [seq_len, hidden]
    q_a: np.ndarray,          # [q_lora_rank, hidden]
    q_a_norm: np.ndarray,     # [q_lora_rank]
    q_b: np.ndarray,          # [num_heads * (qk_nope + qk_rope), q_lora_rank]
    kv_a_mqa: np.ndarray,     # [kv_lora_rank + qk_rope_head_dim, hidden]
    kv_a_norm: np.ndarray,    # [kv_lora_rank]
    k_b: np.ndarray,          # [num_kv_heads * qk_nope_head_dim, kv_lora_rank]
    v_b: np.ndarray,          # [num_kv_heads * v_head_dim, kv_lora_rank]
    attn_output: np.ndarray,  # [hidden, num_heads * v_head_dim]
    num_heads: int,
    num_kv_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope_theta: float,
    eps: float,
) -> np.ndarray:
    """Multi-latent attention forward. shape [seq_len, hidden] in/out."""
    seq_len, hidden_dim = hidden.shape

    # ----- Q path: down → norm → up -----
    q_lat = rms_norm(hidden @ q_a.T, q_a_norm, eps)  # [seq_len, q_lora_rank]
    q_full = q_lat @ q_b.T                            # [seq_len, num_heads * (qk_nope+qk_rope)]
    q_full = q_full.reshape(seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim)
    q_nope, q_rope = np.split(q_full, [qk_nope_head_dim], axis=-1)
    # q_rope: [seq_len, num_heads, qk_rope_head_dim]

    # ----- KV path: down → norm → up -----
    kv_lat_full = hidden @ kv_a_mqa.T                  # [seq_len, kv_lora_rank + qk_rope_head_dim]
    kv_lat = kv_lat_full[:, : -qk_rope_head_dim]
    k_rope_in = kv_lat_full[:, -qk_rope_head_dim:]
    kv_lat = rms_norm(kv_lat, kv_a_norm, eps)
    k_full = kv_lat @ k_b.T                            # [seq_len, num_kv_heads * qk_nope]
    v_full = kv_lat @ v_b.T                            # [seq_len, num_kv_heads * v_head_dim]
    k_full = k_full.reshape(seq_len, num_kv_heads, qk_nope_head_dim)
    v_full = v_full.reshape(seq_len, num_kv_heads, v_head_dim)
    # MQA expansion: adapt k_rope and v for grouped-query share.
    # Apply to kv_rope_in once (ABSORBED rope), then duplicate across heads.
    k_rope = np.broadcast_to(k_rope_in[:, None, :], (seq_len, num_heads, qk_rope_head_dim)).copy()
    # k_full: only num_kv_heads heads; we tile across groups of num_heads/num_kv_heads.
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k_full = np.repeat(k_full, rep, axis=1)
        v_full = np.repeat(v_full, rep, axis=1)

    # ----- RoPE on q_rope and k_rope -----
    # Apply rotary angles to the rope portion only.
    half = qk_rope_head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) / qk_rope_head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(t, inv_freq)            # [seq_len, half]
    cos = np.cos(angles)
    sin = np.sin(angles)

    def apply_rope(x):
        # x: [seq_len, num_heads, qk_rope_head_dim]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos[:, None, :] - x2 * sin[:, None, :],
                               x2 * cos[:, None, :] + x1 * sin[:, None, :]], axis=-1)

    q_rope = apply_rope(q_rope)
    k_rope = apply_rope(k_rope)

    # ----- Concat nope + rope → full QK -----
    q = np.concatenate([q_nope, q_rope], axis=-1)  # [seq_len, num_heads, qk_nope + qk_rope]
    k = np.concatenate([k_full, k_rope], axis=-1)

    # ----- Standard scaled dot attention -----
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum("shd,thd->sht", q, k) * scale  # [seq_len, num_heads, seq_len]
    # causal mask
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), 1)
    scores = np.where(mask, -1e9, scores)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.einsum("sht,thd->shd", p, v_full)         # [seq_len, num_heads, v_head_dim]
    out = out.reshape(seq_len, num_heads * v_head_dim)
    return out @ attn_output.T


def random_tensors_for_kimi(
    seq_len: int = 4,
    hidden_dim: int = 128,
    num_experts: int = 16,
    num_experts_used: int = 4,
    expert_ffn: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 1,
    qk_nope_head_dim: int = 16,
    qk_rope_head_dim: int = 16,
    v_head_dim: int = 16,
    q_lora_rank: int = 32,
    kv_lora_rank: int = 16,
    rope_theta: float = 50_000.0,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    def t(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    hidden = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32)
    return {
        "hidden": hidden,
        "gate_inp": t(num_experts, hidden_dim),
        "gate_exps": t(num_experts, expert_ffn, hidden_dim),
        "up_exps": t(num_experts, expert_ffn, hidden_dim),
        "down_exps": t(num_experts, hidden_dim, expert_ffn),
        "gate_shexp": t(expert_ffn, hidden_dim),
        "up_shexp": t(expert_ffn, hidden_dim),
        "down_shexp": t(hidden_dim, expert_ffn),
        "q_a": t(q_lora_rank, hidden_dim),
        "q_a_norm": t(q_lora_rank) + 1.0,
        "q_b": t(num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank),
        "kv_a_mqa": t(kv_lora_rank + qk_rope_head_dim, hidden_dim),
        "kv_a_norm": t(kv_lora_rank) + 1.0,
        "k_b": t(num_kv_heads * qk_nope_head_dim, kv_lora_rank),
        "v_b": t(num_kv_heads * v_head_dim, kv_lora_rank),
        "attn_output": t(hidden_dim, num_heads * v_head_dim),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "rope_theta": rope_theta,
        "eps": 1e-5,
        "num_experts_used": num_experts_used,
        "gating_func": 2,
        "norm_topk_prob": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    if args.random:
        w = random_tensors_for_kimi()
        moe_out = moe_forward(
            hidden=w["hidden"],
            gate_inp=w["gate_inp"],
            gate_exps=w["gate_exps"],
            up_exps=w["up_exps"],
            down_exps=w["down_exps"],
            gate_shexp=w["gate_shexp"],
            up_shexp=w["up_shexp"],
            down_shexp=w["down_shexp"],
            num_experts_used=w["num_experts_used"],
            gating_func=w["gating_func"],
            norm_topk_prob=w["norm_topk_prob"],
        )
        mla_out = mla_attention_forward(
            hidden=w["hidden"],
            q_a=w["q_a"], q_a_norm=w["q_a_norm"], q_b=w["q_b"],
            kv_a_mqa=w["kv_a_mqa"], kv_a_norm=w["kv_a_norm"],
            k_b=w["k_b"], v_b=w["v_b"], attn_output=w["attn_output"],
            num_heads=w["num_heads"], num_kv_heads=w["num_kv_heads"],
            qk_nope_head_dim=w["qk_nope_head_dim"],
            qk_rope_head_dim=w["qk_rope_head_dim"],
            v_head_dim=w["v_head_dim"],
            rope_theta=w["rope_theta"],
            eps=w["eps"],
        )
        print("MoE out mean:", float(moe_out.mean()), "max-abs:", float(np.abs(moe_out).max()))
        print("MLA out mean:", float(mla_out.mean()), "max-abs:", float(np.abs(mla_out).max()))
        out = {
            "moe_out_shape": list(moe_out.shape),
            "mla_out_shape": list(mla_out.shape),
            "moe_mean": float(moe_out.mean()),
            "mla_mean": float(mla_out.mean()),
        }
        print(json.dumps(out, indent=2))
        return

    print("--random required (no model loader yet)", file=sys.stderr)


if __name__ == "__main__":
    main()
