#!/bin/bash
# Build llama.cpp shared libraries from vendored source
# Usage: ./scripts/build_llama_cpp.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../rust/llama.cpp"
BUILD_DIR="${LLAMA_DIR}/build"

echo "🔨 Building llama.cpp from submodule..."
echo "   Source: ${LLAMA_DIR}"
echo "   Build:  ${BUILD_DIR}"

if [ ! -f "${LLAMA_DIR}/CMakeLists.txt" ]; then
    echo "❌ llama.cpp source not found at ${LLAMA_DIR}"
    exit 1
fi

# Ensure cmake is available
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install cmake"
    echo "   macOS:         brew install cmake"
    exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${LLAMA_DIR}" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_ALL_WARNINGS=OFF \
    -DCMAKE_BUILD_TYPE=Release

JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "   Building with ${JOBS} parallel jobs..."
cmake --build . --parallel "${JOBS}"

echo ""
echo "✅ llama.cpp built successfully!"
echo "   Libraries: ${BUILD_DIR}/bin/"
echo ""
echo "Now build Leafcutter with FFI support:"
echo "   cd rust && cargo build --release --features llama-ffi"
