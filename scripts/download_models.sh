#!/bin/bash
# download_models.sh - Download all test models

set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Please install it: pip install huggingface-hub"
    exit 1
fi

echo "📥 Downloading Tier 1 models (baseline)..."
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir "$MODELS_DIR/tinyllama-1.1b"

echo "✅ TinyLlama downloaded"

echo "📥 Downloading Qwen2-0.5B..."
huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "$MODELS_DIR/qwen2-0.5b"

echo "✅ Qwen2-0.5B downloaded"

echo ""
echo "⚠️  For Mistral-7B-Q4 (4.3GB), download manually:"
echo ""
echo "   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo "   -O $MODELS_DIR/mistral-7b-q4.gguf"
echo ""
echo "✅ Model downloads complete"
