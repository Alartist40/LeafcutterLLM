#!/bin/sh
# install.sh — one-line install for LeafcutterLLM
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | sh
#
# Installs the `leafcutter` binary to /usr/local/bin (or ~/.local/bin as fallback).
# Requires: curl, git, rust/cargo, gcc.

set -e

REPO="https://github.com/Alartist40/LeafcutterLLM.git"
INSTALL_DIR="${HOME}/.leafcutter"
BIN_NAME="leafcutter"

# Determine install location
if [ -w "/usr/local/bin" ]; then
    BIN_DIR="/usr/local/bin"
elif [ -w "${HOME}/.local/bin" ]; then
    BIN_DIR="${HOME}/.local/bin"
else
    mkdir -p "${HOME}/.local/bin"
    BIN_DIR="${HOME}/.local/bin"
fi

echo "LeafcutterLLM — installing to ${BIN_DIR}/${BIN_NAME}"

# Check dependencies
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required. Install it first."
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: Rust/Cargo is required. Install it with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Clone or update
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "Updating existing clone at ${INSTALL_DIR}..."
    cd "${INSTALL_DIR}"
    git pull --rebase
else
    echo "Cloning LeafcutterLLM to ${INSTALL_DIR}..."
    git clone --depth 1 "${REPO}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
fi

# Build (native only, no FFI needed for the core chat REPL)
echo "Building leafcutter (this may take a few minutes)..."
cd "${INSTALL_DIR}/rust"
cargo build --release --no-default-features --bin leafcutter

# Install binary
echo "Linking ${BIN_NAME} to ${BIN_DIR}/..."
cp "${INSTALL_DIR}/rust/target/release/leafcutter" "${BIN_DIR}/leafcutter"
chmod +x "${BIN_DIR}/leafcutter"

# Verify
if command -v leafcutter >/dev/null 2>&1; then
    echo ""
    echo "Done! LeafcutterLLM is installed."
    echo ""
    echo "Quick start:"
    echo "  leafcutter list              # list available models"
    echo "  leafcutter source add <dir>  # point at a folder of models (persisted)"
    echo "  leafcutter run <model>       # start chatting"
    echo "  leafcutter serve             # start HTTP API server"
    echo ""
    echo "Models auto-detect from ./models, ~/Downloads/models, and any /source dirs."
    echo "Or set LEAF_MODELS_DIR=/path/to/models"
else
    echo ""
    echo "Binary installed to ${BIN_DIR}/leafcutter"
    echo "Add ${BIN_DIR} to your PATH if not already there."
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
fi
