#!/bin/sh
# install.sh — one-line install for LeafcutterLLM
#
#   curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | sh
#
# Like `curl ... | sh` for Ollama: it downloads the prebuilt leafcutter
# binary for your OS/CPU, puts it on PATH, and is ready to chat immediately.
# If no prebuilt binary exists yet for your platform (or you pass
# LEAFCUTTER_SOURCE_BUILD=1), it falls back to building from source,
# auto-installing Rust/Cargo and gcc if needed.
#
# Models are found automatically in ./models, ~/Downloads/models, or any
# directory you add later with:  leafcutter source add <dir>
# Or point once up front with:   LEAF_MODELS_DIR=/path/to/models

set -e

REPO="Alartist40/LeafcutterLLM"
BASE_URL="https://github.com/${REPO}"

echo "🌿 LeafcutterLLM — installer"
echo ""

# ---------------------------------------------------------------- platform --
uname_os() {
    os=$(uname -s)
    case "$os" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT) echo "windows" ;;
        *) echo "linux" ;;
    esac
}

uname_arch() {
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64)  echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "$arch" ;;
    esac
}

OS=$(uname_os)
ARCH=$(uname_arch)
echo "Detected: ${OS}/${ARCH}"

EXT=""
[ "$OS" = "windows" ] && EXT=".exe"

# ------------------------------------------------------------- bin location --
# Prefer /usr/local/bin (root, or a homebrew-style setup); fall back to
# ~/.local/bin when not writable.
if [ -w "/usr/local/bin" ]; then
    BIN_DIR="/usr/local/bin"
elif [ -w "${HOME}/.local/bin" ]; then
    BIN_DIR="${HOME}/.local/bin"
else
    mkdir -p "${HOME}/.local/bin"
    BIN_DIR="${HOME}/.local/bin"
fi

BIN_PATH="${BIN_DIR}/leafcutter"
if [ -e "${BIN_PATH}" ]; then
    echo "Existing install found at ${BIN_PATH} — replacing it."
fi

# ----------------------------------------------------------- try download ---
download() {
    asset="leafcutter-${OS}-${ARCH}${EXT}"
    url="${BASE_URL}/releases/latest/download/${asset}"

    echo "Downloading ${asset} …"
    echo "  ${url}"

    tmp="${TMPDIR:-/tmp}/leafcutter-${OS}-${ARCH}"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$tmp" "$url"
    else
        echo "ERROR: need curl or wget to download the binary."
        return 1
    fi

    if [ ! -s "$tmp" ]; then
        echo "Download produced an empty file."
        return 1
    fi

    # Best-effort sanity check: it should be a real executable.
    # ELF    → 7f 45 4c 46
    # PE     → 4d 5a
    # Mach-O → cf fa ed fe / ca fe ba be / fe ed fa ce
    if ! file "$tmp" | grep -qiE "ELF|Mach-O|PE32" 2>/dev/null; then
        if ! head -c 4 "$tmp" | od -An -tx1 | grep -qE "7f 45 4c 46|4d 5a|fa ed fe|fe ed fa"; then
            echo "Downloaded file does not look like a binary — giving up."
            rm -f "$tmp"
            return 1
        fi
    fi

    chmod +x "$tmp"
    mv "$tmp" "${BIN_PATH}"
    echo "Installed to ${BIN_PATH}"
}

if [ "${LEAFCUTTER_SOURCE_BUILD}" != "1" ]; then
    if download; then
        INSTALLED=prebuilt
    else
        echo ""
        echo "No prebuilt binary for ${OS}/${ARCH} yet — building from source."
    fi
fi

# ----------------------------------------------------------- source build ---
# Only reached when the download path failed or was skipped.
if [ "${INSTALLED:-}" != "prebuilt" ]; then
    # --- dependencies ---
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git is required. Install it first."
        exit 1
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        echo "Rust/Cargo not found — installing via rustup (non-interactive)…"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
        else
            echo "ERROR: need curl or wget to install Rust."
            exit 1
        fi
        . "${HOME}/.cargo/env"
    fi

    if ! command -v cc >/dev/null 2>&1; then
        echo "WARNING: no C compiler found. Building may fail."
        echo "  Debian/Ubuntu:  sudo apt install build-essential"
        echo "  macOS:          xcode-select --install"
        echo "  Fedora:         sudo dnf install gcc gcc-c++"
    fi

    # --- clone or update ---
    INSTALL_DIR="${HOME}/.leafcutter"
    if [ -d "${INSTALL_DIR}/.git" ]; then
        echo "Updating existing clone at ${INSTALL_DIR}…"
        (cd "${INSTALL_DIR}" && git pull --rebase)
    else
        echo "Cloning LeafcutterLLM to ${INSTALL_DIR}…"
        git clone --depth 1 "${BASE_URL}.git" "${INSTALL_DIR}"
    fi

    # --- build ---
    echo "Building leafcutter (release, may take a few minutes)…"
    (cd "${INSTALL_DIR}/rust" && cargo build --release --bin leafcutter)

    # --- install ---
    cp "${INSTALL_DIR}/rust/target/release/leafcutter" "${BIN_PATH}"
    chmod +x "${BIN_PATH}"
    echo "Installed to ${BIN_PATH}"
fi

# ------------------------------------------------------------------ models ---
# Quietly verify the binary runs (don't print "command not found" noise when
# PATH hasn't been refreshed yet in this shell).
"$BIN_PATH" list >/dev/null 2>&1 || true

echo ""
echo "✅ LeafcutterLLM installed."
echo ""
echo "Quick start:"
echo "  leafcutter list              # list available models"
echo "  leafcutter source add <dir>  # point at a folder of models (persisted)"
echo "  leafcutter run <model>       # start chatting"
echo "  leafcutter serve             # start HTTP API server"
echo ""
echo "Models auto-detect from ./models, ~/Downloads/models, and any /source dirs."
echo "Or set LEAF_MODELS_DIR=/path/to/models"
if [ "${BIN_DIR}" != "/usr/local/bin" ]; then
    echo ""
    echo "Add ${BIN_DIR} to your PATH if not already there:"
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
fi
