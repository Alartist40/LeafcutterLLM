# 📥 Download Models

Here are some recommended models to get started with LeafcutterLLM.

> **Tip (2026-08-01):** Ornith 1.0 9B (Qwen3.5 hybrid) is the fully verified
> flagship — run `leafcutter run ornith` with the Q4_K_M GGUF for coherent
> native chat at ~8.1 GB peak RAM. GGUF files are the recommended format.

## Ornith 1.0 9B (Qwen3.5 hybrid) — flagship, native
*Qwen 3.5 hybrid (DeltaNet linear attention + full attention interleaved)*
- Ornith-1.0-9b Q4_K_M (5.3 GB) — verified: `leafcutter run ornith`
- Ornith-1.0-9b Q6_K (7.4 GB) — verified forward + coherent generation

## TinyLlama (1.1B Parameters)
*Great for Raspberry Pi Zero 2W or Pi 3*
- [TinyLlama-1.1B-Chat-v1.0 (GGUF Q4_K_M)](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

## Qwen2 (1.5B Parameters)
*High performance for low RAM*
- [Qwen2-1.5B-Instruct (GGUF Q4_K_M)](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF)

## LLaMA-3 (8B Parameters)
*Standard for Pi 5 (8GB) or Laptops*
- [Meta-Llama-3-8B-Instruct (GGUF Q4_K_M)](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF)

## Phi-3 Mini (3.8B Parameters)
*Microsoft's highly capable small model*
- [Phi-3-mini-4k-instruct (GGUF Q4_K_M)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
