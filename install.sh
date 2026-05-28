#!/usr/bin/env bash
# LeafcutterLLM — One-Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | bash
set -euo pipefail

LEAFCUTTER_VERSION="0.9.0"
REPO_URL="https://github.com/Alartist40/LeafcutterLLM.git"
INSTALL_DIR="${HOME}/.leafcutter"
BIN_DIR="${HOME}/.local/bin"
LLAMA_CPP_DIR="${INSTALL_DIR}/LeafcutterLLM/rust/llama.cpp"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}ℹ️  $1${NC}"; }
ok()    { echo -e "${GREEN}✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()   { echo -e "${RED}❌ $1${NC}"; }

# ─── Detect OS ───────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected: $OS ($ARCH)"

# ─── Check prerequisites ─────────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

info "Checking prerequisites..."

MISSING=""
for cmd in git cmake make curl; do
    if ! check_cmd "$cmd"; then
        MISSING="$MISSING $cmd"
    fi
done

if [ -n "$MISSING" ]; then
    err "Missing required tools:$MISSING"
    echo "Please install them first:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y git cmake make curl build-essential"
    echo "  macOS:         xcode-select --install && brew install cmake git curl"
    echo "  Fedora:        sudo dnf install -y git cmake make curl gcc-c++"
    exit 1
fi

# Check for Rust
if ! check_cmd rustc; then
    warn "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "${HOME}/.cargo/env"
    ok "Rust installed: $(rustc --version)"
else
    ok "Rust found: $(rustc --version)"
fi

# Check for clang (needed by bindgen / some builds)
if ! check_cmd clang; then
    warn "clang not found. Some builds may fail."
    echo "  Ubuntu/Debian: sudo apt install -y clang"
fi

# ─── Setup directories ───────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# ─── Clone or update Leafcutter ──────────────────────────────────────────────
if [ -d "${INSTALL_DIR}/LeafcutterLLM/.git" ]; then
    info "Updating LeafcutterLLM..."
    cd "${INSTALL_DIR}/LeafcutterLLM"
    git pull --ff-only origin main
else
    info "Cloning LeafcutterLLM..."
    git clone --depth 1 "$REPO_URL" "${INSTALL_DIR}/LeafcutterLLM"
    cd "${INSTALL_DIR}/LeafcutterLLM"
fi
ok "LeafcutterLLM source ready"

# ─── Build vendored llama.cpp shared libraries ───────────────────────────────
info "Building llama.cpp shared libraries..."
cd "$LLAMA_CPP_DIR"

# Clean previous build to ensure ABI compatibility
rm -rf build
mkdir -p build && cd build

cmake .. \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_ALL_WARNINGS=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Determine parallel build flag
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
info "Building with $JOBS parallel jobs..."
cmake --build . --parallel "$JOBS"

ok "llama.cpp libraries built in ${LLAMA_CPP_DIR}/build/bin/"

# ─── Build Leafcutter ────────────────────────────────────────────────────────
info "Building Leafcutter v${LEAFCUTTER_VERSION}..."
cd "${INSTALL_DIR}/LeafcutterLLM/rust"

export LD_LIBRARY_PATH="${LLAMA_CPP_DIR}/build/bin:${LD_LIBRARY_PATH:-}"

cargo build --release --bin leafcutter

ok "Leafcutter built successfully"

# ─── Install leafcutter binary ───────────────────────────────────────────────
info "Installing leafcutter to ${BIN_DIR}..."
cp "${INSTALL_DIR}/LeafcutterLLM/rust/target/release/leafcutter" "${BIN_DIR}/leafcutter"
chmod +x "${BIN_DIR}/leafcutter"

# ─── Shell profile setup ─────────────────────────────────────────────────────
SHELL_PROFILE=""
if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_PROFILE="${HOME}/.zshrc"
elif [ -n "${BASH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "bash" ]; then
    SHELL_PROFILE="${HOME}/.bashrc"
else
    SHELL_PROFILE="${HOME}/.profile"
fi

# Add BIN_DIR to PATH if not already there
if ! grep -q "$BIN_DIR" "$SHELL_PROFILE" 2>/dev/null; then
    echo "" >> "$SHELL_PROFILE"
    echo "# LeafcutterLLM" >> "$SHELL_PROFILE"
    echo 'export PATH="'"$BIN_DIR"':${PATH}"' >> "$SHELL_PROFILE"
    ok "Added ${BIN_DIR} to PATH in ${SHELL_PROFILE}"
fi

# Add LD_LIBRARY_PATH for llama.cpp if not already there
LLAMA_LIB_LINE="export LD_LIBRARY_PATH=\"${LLAMA_CPP_DIR}/build/bin:\${LD_LIBRARY_PATH:-}\""
if ! grep -q "${LLAMA_CPP_DIR}/build/bin" "$SHELL_PROFILE" 2>/dev/null; then
    echo "$LLAMA_LIB_LINE" >> "$SHELL_PROFILE"
    ok "Added llama.cpp library path to ${SHELL_PROFILE}"
fi

# ─── Create wrapper script for immediate use ─────────────────────────────────
WRAPPER="${BIN_DIR}/leafcutter"
cat > "$WRAPPER" << EOF
#!/usr/bin/env bash
# Leafcutter wrapper — ensures LD_LIBRARY_PATH is set
export LD_LIBRARY_PATH="${LLAMA_CPP_DIR}/build/bin:\${LD_LIBRARY_PATH:-}"
exec "${INSTALL_DIR}/LeafcutterLLM/rust/target/release/leafcutter" "\$@"
EOF
chmod +x "$WRAPPER"

# ─── Verification ────────────────────────────────────────────────────────────
info "Verifying installation..."
export LD_LIBRARY_PATH="${LLAMA_CPP_DIR}/build/bin:${LD_LIBRARY_PATH:-}"

if "${BIN_DIR}/leafcutter" --version &> /dev/null; then
    ok "leafcutter is installed and working!"
    "${BIN_DIR}/leafcutter" --version
else
    err "Installation verification failed."
    exit 1
fi

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🌿 LeafcutterLLM v${LEAFCUTTER_VERSION} Installed!          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
ok "Installation complete!"
echo ""
echo "Quick start:"
echo "  1. Reload your shell:   source ${SHELL_PROFILE}"
echo "  2. List models:         leafcutter list-models --dir ~/models"
echo "  3. Generate text:       leafcutter generate --model model.gguf --prompt 'Hello'"
echo "  4. Chat interactively:  leafcutter chat --model model.gguf"
echo "  5. Start API server:    leafcutter server --model model.gguf --port 8081"
echo ""
echo "Download models with Cynapse:"
echo "  cynapse model download meta-llama/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf"
echo ""
echo "Or manually from HuggingFace:"
echo "  wget https://huggingface.co/.../resolve/main/model.gguf -P ~/models/"
echo ""
