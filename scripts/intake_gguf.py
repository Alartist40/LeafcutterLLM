#!/usr/bin/env python3
"""
LeafcutterLLM — GGUF intake checklist
======================================

For each GGUF model, prints an architecture-intake summary that is the
*source of truth* for whether the engine can run it natively, what kernel
path it needs (dense / MoE / MLA / sparse-indexed), and how much RAM a
layer-streaming resident peak should be expected to hit.

Usage:
    python3 scripts/intake_gguf.py --model PATH [PATH ...]
    python3 scripts/intake_gguf.py --dir /mnt/ssd/Xander/AI\\ Models/

It also produces a structured JSON to stdout (one record per model) when
called with `--json`, useful for diffing across releases.
"""

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, asdict
from typing import Any


GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ2_S = 18
GGML_TYPE_IQ3_XXS = 19
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ1_S = 24
GGML_TYPE_IQ4_NL = 25

QUANT_NAME = {
    GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0", GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0", GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0", GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K", GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K", GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K", GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS", GGML_TYPE_IQ2_XS: "IQ2_XS",
    GGML_TYPE_IQ2_S: "IQ2_S",
    GGML_TYPE_IQ3_XXS: "IQ3_XXS", GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ4_NL: "IQ4_NL",
}

# What Leafcutter's native kernels actually dequantize today.
NATIVE_QUANTS = {
    GGML_TYPE_F32, GGML_TYPE_F16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
    GGML_TYPE_IQ4_NL,
}


@dataclass
class IntakeReport:
    path: str
    size_gb: float
    arch: str
    family: str                # dense | deepseek | glm-dsa | qwen_hybrid | unknown
    capabilities: list[str]    # ["dense", "mla", "moe", "shared_expert", "mtp", "rope_yarn", "indexed_attn"]
    quant_summary: dict[str, Any]
    dims: dict[str, Any]
    expected_per_layer_resident_mb: float
    native_support: str         # "native" | "research" | "needs_arch_enum" | "bridge-only"
    notes: list[str]


def read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    if n > 50_000_000 or n < 0:
        return f"<big n={n}>"
    return f.read(n).decode("utf-8", "replace")


def parse_metadata(path: str) -> dict:
    """Parse GGUF key/value pairs from the header shard."""
    kv = {}
    n_kv = 0
    n_tensor = 0
    version = 0
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            return {"_error": f"not a GGUF file (magic={magic!r})"}
        version = struct.unpack("<I", f.read(4))[0]
        n_tensor = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        pos0 = f.tell()
        for _ in range(n_kv):
            if f.tell() - pos0 > 1_000_000:
                kv["_stopped_at_bytecap"] = True
                break
            try:
                k = read_str(f)
                if k.startswith("<big"):
                    break
                t = struct.unpack("<I", f.read(4))[0]
                if t == 8:  # STRING
                    kv[k] = read_str(f)
                elif t == 4:
                    kv[k] = struct.unpack("<I", f.read(4))[0]
                elif t == 10:
                    kv[k] = struct.unpack("<Q", f.read(8))[0]
                elif t == 5:
                    kv[k] = struct.unpack("<i", f.read(4))[0]
                elif t == 6:
                    kv[k] = round(struct.unpack("<f", f.read(4))[0], 6)
                elif t == 12:
                    kv[k] = round(struct.unpack("<d", f.read(8))[0], 6)
                elif t == 7:
                    kv[k] = bool(f.read(1)[0])
                elif t == 9:  # ARRAY — small arrays walked; large ones skipped by element-size * count
                    at = struct.unpack("<I", f.read(4))[0]
                    an = struct.unpack("<Q", f.read(8))[0]
                    if an > 200 and at != 8:
                        # Don't know exact bytes; bail out.
                        kv[k] = f"<arr type={at} len={an}>"
                        kv["_stopped_at_bigarray"] = k
                        break
                    elems = []
                    if an > 200 and at == 8:
                        # walk strings one at a time but don't store
                        for _ in range(an):
                            read_str(f)
                        kv[k] = f"<arr type={at} len={an}>"
                        continue
                    for _ in range(an):
                        if at == 8:
                            elems.append(read_str(f))
                        elif at == 0: elems.append(f.read(1)[0])
                        elif at == 4: elems.append(struct.unpack("<I", f.read(4))[0])
                        elif at == 5: elems.append(struct.unpack("<i", f.read(4))[0])
                        elif at == 6: elems.append(round(struct.unpack("<f", f.read(4))[0], 4))
                        elif at == 7: elems.append(bool(f.read(1)[0]))
                        elif at == 10: elems.append(struct.unpack("<Q", f.read(8))[0])
                        elif at == 11: elems.append(struct.unpack("<q", f.read(8))[0])
                        elif at == 12: elems.append(round(struct.unpack("<d", f.read(8))[0], 4))
                        else:
                            f.read(8)
                            elems.append(f"?{at}")
                    kv[k] = elems
            except Exception as e:
                kv["_error"] = str(e)
                break
    return kv


def collect_tensor_quant_types(path: str) -> dict:
    """Walk all tensor headers across ALL shards (by glob) and collect quant types.

    The header shard only has n_tensor_field=0 for split GGUFs; we search
    for *.gguf files sharing the same naming root and walk them all.
    """
    # For now, we walk just the input file.  For shard 1 the field is 0
    # so we fall back to other shards to learn what quant types are used.
    files_to_walk = [path]
    root = path.rsplit("-", 1)[0]  # "GLM-5.2-UD-Q4_K_XL"
    if root and os.path.isdir(os.path.dirname(path)):
        d = os.path.dirname(path)
        for f in sorted(os.listdir(d)):
            if f.startswith(os.path.basename(root)) and f.endswith(".gguf"):
                files_to_walk.append(os.path.join(d, f))

    quants = {}
    total = 0
    for p in files_to_walk:
        try:
            with open(p, "rb") as f:
                magic = f.read(4)
                if magic != GGUF_MAGIC:
                    continue
                struct.unpack("<I", f.read(4))[0]  # ver
                n_tensor = struct.unpack("<Q", f.read(8))[0]
                n_kv = struct.unpack("<Q", f.read(8))[0]
                # we'll skip the kv block roughly; header shard has 60-69
                # but body shards have 3 (split.*).  Easier: tokenize up to n_kv quickly
                for _ in range(n_kv):
                    try:
                        _skip_meta(f)
                    except Exception:
                        break
                # now read tensor headers
                for i in range(n_tensor):
                    try:
                        tn = read_str(f)
                        if tn.startswith("<big"):
                            break
                        nd = struct.unpack("<I", f.read(4))[0]
                        if nd > 8: break
                        for _ in range(nd):
                            f.read(8)
                        tt = struct.unpack("<I", f.read(4))[0]
                        f.read(8)  # offset
                        name = QUANT_NAME.get(tt, f"?{tt}")
                        quants.setdefault(name, 0)
                        quants[name] += 1
                        total += 1
                    except Exception:
                        break
        except FileNotFoundError:
            pass
    return {"by_quant": quants, "total_tensors": total}


def _skip_meta(f):
    """Skip one kv pair in a GGUF file. Limited robustness."""
    k = read_str(f)
    if k.startswith("<big"):
        raise ValueError("eof in key")
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        read_str(f)
    elif t in (4, 7):  # uint32, bool (bool is 4 bytes? actually 7=bool=1 byte)
        if t == 7:
            f.read(1)
        else:
            f.read(4)
    elif t in (10, 11, 12, 6, 5):  # uint64, int64, double, float, int32
        sz = {5:4, 6:4, 10:8, 11:8, 12:8}.get(t, 4)
        f.read(sz)
    elif t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        an = struct.unpack("<Q", f.read(8))[0]
        if an > 500:
            raise ValueError("skipping array; can't know element size of big string array")
        for _ in range(an):
            if at == 8:
                read_str(f)
            elif at in (5,6,7,4):
                f.read({5:4,6:4,7:1,4:4}.get(at, 4))
            else:
                f.read(8)
    else:
        f.read(8)


def classify_arch(kv: dict) -> tuple[str, str, list[str], dict]:
    arch = kv.get("general.architecture", "<none>")
    size_label = kv.get("general.size_label", "")
    family = "unknown"
    caps: list[str] = []
    dims: dict = {}

    if arch in ("deepseek2", "deepseek"):
        family = "deepseek"
        caps.append("mla")
        caps.append("moe")
        caps.append("shared_expert")
        n_experts = kv.get("deepseek2.expert_count")
        if n_experts and n_experts > 1:
            caps.append("routed_experts")
        if kv.get("deepseek2.rope.scaling.type") == "yarn":
            caps.append("rope_yarn")
        if kv.get("deepseek2.leading_dense_block_count"):
            caps.append(f"leading_dense={kv['deepseek2.leading_dense_block_count']}")
        # MTP expected on Kimi K2.6
        dims.update({
            "n_layer": kv.get("deepseek2.block_count"),
            "hidden": kv.get("deepseek2.embedding_length"),
            "ffn_dim": kv.get("deepseek2.feed_forward_length"),
            "experts": n_experts,
            "experts_used": kv.get("deepseek2.expert_used_count"),
            "shared_experts": kv.get("deepseek2.expert_shared_count"),
            "expert_ffn": kv.get("deepseek2.expert_feed_forward_length"),
            "expert_scale": kv.get("deepseek2.expert_weights_scale"),
            "head_count": kv.get("deepseek2.attention.head_count"),
            "kv_heads": kv.get("deepseek2.attention.head_count_kv"),
            "qk_nope_head_dim": kv.get("deepseek2.attention.key_length_mla"),
            "qk_rope_head_dim": (kv.get("deepseek2.attention.key_length") or 0) - (kv.get("deepseek2.attention.key_length_mla") or 0),
            "v_head_dim": kv.get("deepseek2.attention.value_length_mla"),
            "q_lora_rank": kv.get("deepseek2.attention.q_lora_rank"),
            "kv_lora_rank": kv.get("deepseek2.attention.kv_lora_rank"),
            "rope_dim": kv.get("deepseek2.rope.dimension_count"),
            "rope_theta": kv.get("deepseek2.rope.freq_base"),
            "ctx": kv.get("deepseek2.context_length"),
            "vocab": kv.get("deepseek2.vocab_size"),
        })
    elif arch == "glm-dsa":
        family = "glm-dsa"
        caps.append("mla")
        caps.append("moe")
        caps.append("shared_expert")
        if kv.get("glm-dsa.attention.indexer.top_k"):
            caps.append("indexed_attn")
        if kv.get("glm-dsa.nextn_predict_layers"):
            caps.append(f"mtp={kv['glm-dsa.nextn_predict_layers']}")
        if kv.get("glm-dsa.leading_dense_block_count"):
            caps.append(f"leading_dense={kv['glm-dsa.leading_dense_block_count']}")
        dims.update({
            "n_layer": kv.get("glm-dsa.block_count"),
            "hidden": kv.get("glm-dsa.embedding_length"),
            "ffn_dim": kv.get("glm-dsa.feed_forward_length"),
            "experts": kv.get("glm-dsa.expert_count"),
            "experts_used": kv.get("glm-dsa.expert_used_count"),
            "shared_experts": kv.get("glm-dsa.expert_shared_count"),
            "expert_ffn": kv.get("glm-dsa.expert_feed_forward_length"),
            "expert_scale": kv.get("glm-dsa.expert_weights_scale"),
            "head_count": kv.get("glm-dsa.attention.head_count"),
            "kv_heads": kv.get("glm-dsa.attention.head_count_kv"),
            "qk_nope_head_dim": kv.get("glm-dsa.attention.key_length_mla"),
            "qk_rope_head_dim": (kv.get("glm-dsa.attention.key_length") or 0) - (kv.get("glm-dsa.attention.key_length_mla") or 0),
            "v_head_dim": kv.get("glm-dsa.attention.value_length_mla"),
            "q_lora_rank": kv.get("glm-dsa.attention.q_lora_rank"),
            "kv_lora_rank": kv.get("glm-dsa.attention.kv_lora_rank"),
            "indexer_heads": kv.get("glm-dsa.attention.indexer.head_count"),
            "indexer_top_k": kv.get("glm-dsa.attention.indexer.top_k"),
            "rope_dim": kv.get("glm-dsa.rope.dimension_count"),
            "rope_theta": kv.get("glm-dsa.rope.freq_base"),
            "ctx": kv.get("glm-dsa.context_length"),
            "vocab": kv.get("glm-dsa.vocab_size"),
            "nextn_layers": kv.get("glm-dsa.nextn_predict_layers"),
        })
    elif arch == "llama":
        family = "dense"
        caps.append("dense_attn")
        dims.update({
            "n_layer": kv.get("llama.block_count"),
            "hidden": kv.get("llama.embedding_length"),
            "ffn_dim": kv.get("llama.feed_forward_length"),
            "head_count": kv.get("llama.attention.head_count"),
            "kv_heads": kv.get("llama.attention.head_count_kv"),
            "ctx": kv.get("llama.context_length"),
            "vocab": kv.get("llama.vocab_size"),
            "rope_theta": kv.get("llama.rope.freq_base"),
        })
    elif arch in ("qwen2", "qwen3"):
        family = "dense"
        caps.append("dense_attn")
        prefix = "qwen2"
        if arch == "qwen3": prefix = "qwen3"
        dims.update({
            "n_layer": kv.get(f"{prefix}.block_count"),
            "hidden": kv.get(f"{prefix}.embedding_length"),
            "ffn_dim": kv.get(f"{prefix}.feed_forward_length"),
            "head_count": kv.get(f"{prefix}.attention.head_count"),
            "kv_heads": kv.get(f"{prefix}.attention.head_count_kv"),
            "ctx": kv.get(f"{prefix}.context_length"),
            "vocab": kv.get(f"{prefix}.vocab_size"),
            "rope_theta": kv.get(f"{prefix}.rope.freq_base"),
        })
    elif arch == "qwen35":
        family = "qwen_hybrid"
        caps.extend(["dense_attn", "delta_net"])
        prefix = "qwen35"
        dims.update({
            "n_layer": kv.get(f"{prefix}.block_count"),
            "hidden": kv.get(f"{prefix}.embedding_length"),
            "ffn_dim": kv.get(f"{prefix}.feed_forward_length"),
            "head_count": kv.get(f"{prefix}.attention.head_count"),
            "kv_heads": kv.get(f"{prefix}.attention.head_count_kv"),
            "ctx": kv.get(f"{prefix}.context_length"),
            "vocab": kv.get(f"{prefix}.vocab_size"),
        })
    return family, arch, caps, dims


def estimate_resident_per_layer_mb(family: str, dims: dict) -> float:
    """Estimate RSS peak per layer if we hold *one* layer resident."""
    if family == "deepseek" or family == "glm-dsa":
        hidden = int(dims.get("hidden") or 7168)
        q_lora = int(dims.get("q_lora_rank") or 512)
        kv_lora = int(dims.get("kv_lora_rank") or 512)
        n_heads = int(dims.get("head_count") or 64)
        kv_heads = int(dims.get("kv_heads") or 1)
        qk_nope = int(dims.get("qk_nope_head_dim") or 192)
        qk_rope = int(dims.get("qk_rope_head_dim") or 64)
        rope_dim = int(dims.get("rope_dim") or (qk_rope + qk_nope - 576) if False else 64)
        v_head = int(dims.get("v_head_dim") or 128)
        # Approximate MLA-attention-resident bytes (Q4_K_XL ≈ 4.5 b/param)
        attn_bytes = (
            hidden * q_lora
            + q_lora * n_heads * (qk_rope + qk_nope)
            + hidden * (kv_lora + qk_rope)  # kv_a_mqa absorbs rope
            + kv_lora * qk_nope * kv_heads
            + kv_lora * v_head * kv_heads
            + n_heads * v_head * hidden
        )
        return round(attn_bytes * 4.5 / 8 / 1_000_000, 1)
    elif family == "dense":
        hidden = int(dims.get("hidden") or 4096)
        ffn = int(dims.get("ffn_dim") or 14336)
        return round((3 * hidden * ffn + 4 * hidden * hidden + hidden * hidden) * 4.5 / 8 / 1_000_000, 1)
    else:
        return 0.0


def native_support_level(family: str, caps: list[str], dims: dict) -> tuple[str, list[str]]:
    if family in ("dense", "qwen_hybrid"):
        return "native", []
    if family in ("deepseek", "glm-dsa"):
        notes = ["requires arch enum + MoE forward + MLA forward"]
        if "indexed_attn" in caps:
            notes.append("requires sparse attention indexer (GLM-DSA feature)")
        if "mtp" in " ".join(caps):
            notes.append("requires MTP verification")
        return "research", notes
    return "unsupported", [f"unknown architecture: {family}"]


def intake_one(path: str) -> IntakeReport:
    size_gb = os.path.getsize(path) / 1e9
    kv = parse_metadata(path)
    family, arch, caps, dims = classify_arch(kv)
    quants = collect_tensor_quant_types(path)

    expected_mb = estimate_resident_per_layer_mb(family, dims)
    support, notes = native_support_level(family, caps, dims)

    return IntakeReport(
        path=path,
        size_gb=round(size_gb, 3),
        arch=arch,
        family=family,
        capabilities=caps,
        quant_summary=quants,
        dims=dims,
        expected_per_layer_resident_mb=expected_mb,
        native_support=support,
        notes=notes,
    )


def fmt_report(r: IntakeReport) -> str:
    lines = []
    lines.append(f"\n{'-' * 78}")
    lines.append(f"MODEL: {os.path.basename(r.path)}")
    lines.append(f"  size:        {r.size_gb:.3f} GB (all shards combined)")
    lines.append(f"  arch:        {r.arch}  (family={r.family})")
    lines.append(f"  support:     {r.native_support}")
    if r.notes:
        for n in r.notes:
            lines.append(f"               - {n}")
    lines.append(f"  caps:        {', '.join(r.capabilities) or '<none>'}")
    lines.append(f"  dims:")
    for k, v in sorted(r.dims.items(), key=lambda kv: kv[0]):
        lines.append(f"               {k:30s} = {v}")
    lines.append(f"  quants: {r.quant_summary}")
    if r.expected_per_layer_resident_mb:
        lines.append(f"  ~resident per layer (rough est, Q4_K_XL):  {r.expected_per_layer_resident_mb} MB")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", help="Path to GGUF model (can repeat)")
    p.add_argument("--dir", help="Directory to scan for *.gguf recursively")
    p.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = p.parse_args()

    targets = []
    if args.model:
        targets.extend(args.model)
    if args.dir:
        for d, _, files in os.walk(args.dir):
            for f in sorted(files):
                if f.endswith(".gguf"):
                    fpath = os.path.join(d, f)
                    # Avoid the duplicate "tail shard" entry — only include shard 1
                    if fpath not in targets and ("-00001-" in fpath or fpath not in [t for t in targets]):
                        targets.append(fpath)

    if not targets:
        p.print_help()
        sys.exit(1)

    reports = [intake_one(t) for t in targets]
    if args.json:
        out = [asdict(r) for r in reports]
        # Path strings only
        sys.stdout.write(json.dumps(out, indent=2, default=str) + "\n")
    else:
        for r in reports:
            sys.stdout.write(fmt_report(r) + "\n")


if __name__ == "__main__":
    main()
