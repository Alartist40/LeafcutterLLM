use std::path::Path;

fn main() {
    // Only link llama.cpp when the FFI feature is actually enabled. Without
    // this guard a vendored `llama.cpp/build` directory (or an x86_64 build
    // on this machine) would inject host-architecture .so link directives
    // into a cross-compiled or aarch64 build.
    #[cfg(feature = "llama-ffi")]
    {
        link_llama_cpp();
    }
}

#[cfg(feature = "llama-ffi")]
fn link_llama_cpp() {
    let llama_build = std::env::var("LLAMA_CPP_BUILD")
        .unwrap_or_else(|_| {
            // Check vendored path first
            let vendored_build = Path::new("llama.cpp/build");
            if vendored_build.exists() {
                return "llama.cpp/build".to_string();
            }
            // Fallback: Ollama bundles libllama.so at a stable path on Linux.
            // This lets us use llama.cpp as ground truth without building it.
            let ollama_lib = Path::new("/usr/local/lib/ollama");
            if ollama_lib.join("libllama.so").exists() {
                return ollama_lib.to_string_lossy().to_string();
            }
            // Final fallback to the old hardcoded path for backwards compat
            "/home/xander/Documents/llama.cpp/build".to_string()
        });

    let lib_dir = format!("{}/bin", llama_build);
    // Ollama bundles the .so directly in the lib dir (not /bin/),
    // so we accept either layout.
    let so = format!("{}/bin/libllama.so", llama_build);
    let so_alt = format!("{}/libllama.so", llama_build);
    let dylib = format!("{}/bin/libllama.dylib", llama_build);
    let dll = format!("{}/bin/llama.dll", llama_build);

    let lib_exists = Path::new(&so).exists()
        || Path::new(&so_alt).exists()
        || Path::new(&dylib).exists()
        || Path::new(&dll).exists();

    // Pick the directory that actually contains the .so.
    let effective_lib_dir = if Path::new(&so).exists() {
        format!("{}/bin", llama_build)
    } else {
        llama_build.clone()
    };

    if lib_exists {
        println!("cargo:rustc-link-search=native={}", effective_lib_dir);
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", effective_lib_dir);
        println!("cargo:warning=llama.cpp linked from {}", effective_lib_dir);
    } else {
        println!("cargo:warning=llama.cpp not found at {}.", llama_build);
        println!("cargo:warning=  To enable FFI fallback for Qwen3.5/3.6 and exotic quants:");
        println!("cargo:warning=    1. ./scripts/build_llama_cpp.sh");
        println!("cargo:warning=    2. cd rust && cargo build --release --features llama-ffi");
        println!("cargo:warning=  Or set LLAMA_CPP_BUILD=/path/to/llama.cpp/build");
    }
}
