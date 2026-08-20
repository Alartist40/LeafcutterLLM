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
    # Create symlink 'leaf' -> 'leafcutter' for convenience
    ln -sf "${BIN_PATH}" "${BIN_DIR}/leaf"
    echo "Installed to ${BIN_PATH} (and symlinked ${BIN_DIR}/leaf)"
fi

# ------------------------------------------------------------------ models ---
# Quietly verify the binary runs (don't print "command not found" noise when
# PATH hasn't been refreshed yet in this shell).
"$BIN_PATH" list >/dev/null 2>&1 || true

# -------------------------------------------------------------- settings ---
# Write a documented settings template so the shipped binary's env vars and
# defaults are captured on every install (rebuild-from-scratch friendly).
CFG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/leafcutter"
mkdir -p "${CFG_DIR}"
SETTINGS_EXAMPLE="${CFG_DIR}/settings.env.example"
cat > "${SETTINGS_EXAMPLE}" <<'SETTINGS'
# LeafcutterLLM — settings template (copy to settings.env and export as needed)
# Every variable below is OPTIONAL; the binary has sane built-in defaults.

# --- model search ---
# LEAF_MODELS_DIR=/models:/more/models        # colon-separated extra model dirs
#                                              # (default: ./models, ~/Downloads/models)

# --- performance ---
# LEAFCUTTER_THREADS=12                        # Rayon worker threads
#                                              # (default: physical cores on ARM, cores-1 on x86)
# LEAFCUTTER_NO_CACHE=1                        # Tier 3 low-RAM mode: stream + evict layers
# LEAFCUTTER_PREFETCH=1                        # prefetch layer l+1 during layer l (default on)
# LEAFCUTTER_CACHE_MB=6000                     # resident layer cache ceiling
# LEAFCUTTER_CTX_KB=4096                       # KV-cache context budget
# LEAFCUTTER_Q8_GEMV=0                         # 0 disables ARM sdot Q8 GEMV (NEON fallback)
# LEAFCUTTER_PREFER_GPU=1                      # prefer Vulkan/GPU path when available

# --- determinism / debugging ---
# LEAFCUTTER_DETERMINISTIC=1                   # bit-identical serial reductions
# LEAFCUTTER_PROFILE=1                         # per-component layer timing to stderr
# LEAFCUTTER_DEBUG=1                           # verbose debug output
# LEAFCUTTER_DEBUG_LAYERS=0,5,10               # per-layer tensor debug
# LEAFCUTTER_DEBUG_NORMS=1                     # rms-norm diagnostics
# LEAFCUTTER_DEBUG_PROMPT=1                    # print the exact prompt sent to the model
# LEAFCUTTER_ROPE_DEBUG=1                      # RoPE angle diagnostics
# LEAFCUTTER_TOKENIZER_DEBUG=1                 # tokenizer decode diagnostics
# LEAFCUTTER_DELTANET_DEBUG=1                  # DeltaNet layer diagnostics
# LEAFCUTTER_CHUNK_DEBUG=1                     # GGUF chunk parser diagnostics
# LEAFCUTTER_OLLAMA_DEBUG=1                    # Ollama API adapter diagnostics
# LEAFCUTTER_CPU_MONITOR=1                     # log CPU temp / throttling warnings

# --- sampling / API ---
# LEAFCUTTER_TOP_K=2048                        # sampling top-k cap
# LEAFCUTTER_API_KEY=sk-...                    # remote model API key (API-backed models)
# LEAFCUTTER_BASE_URL=http://localhost:11434   # remote model API base URL
# LEAFCUTTER_MODEL=/path/to/model.gguf         # model path override for scripts
SETTINGS
echo "Settings template:        ${SETTINGS_EXAMPLE}"

echo ""
echo "✅ LeafcutterLLM installed successfully."
echo ""
echo "Quick start (using 'leafcutter' or shortcut 'leaf'):"
echo "  leafcutter list                # list available models"
echo "  leafcutter source add <dir>    # point at a folder of models (persisted)"
echo "  leafcutter run <model>         # start chatting (gold & purple REPL UI)"
echo "  leafcutter generate --model .. # one-shot generation"
echo "  leafcutter serve               # start OpenAI-compatible HTTP API server"
echo "  leafcutter update              # update to the latest version"
echo ""
echo "Models auto-detect from ./models, ~/Downloads/models, and any /source dirs."
echo "Or set LEAF_MODELS_DIR=/path/to/models"
echo "Tune runtime behavior with env vars (see ${SETTINGS_EXAMPLE})."
if [ "${BIN_DIR}" != "/usr/local/bin" ]; then
    echo ""
    echo "Add ${BIN_DIR} to your PATH if not already there:"
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
fi

