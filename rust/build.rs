use std::path::Path;

fn main() {
    // Allow overriding the llama.cpp build path via env var.
    let llama_build = std::env::var("LLAMA_CPP_BUILD")
        .unwrap_or_else(|_| {
            // Check vendored path first
            let vendored_build = Path::new("llama.cpp/build");
            if vendored_build.exists() {
                return "llama.cpp/build".to_string();
            }
            // Fallback to the old hardcoded path for backwards compat
            "/home/xander/Documents/llama.cpp/build".to_string()
        });

    let lib_dir = format!("{}/bin", llama_build);
    let so = format!("{}/libllama.so", lib_dir);
    let dylib = format!("{}/libllama.dylib", lib_dir);
    let dll = format!("{}/llama.dll", lib_dir);

    let lib_exists = Path::new(&so).exists()
        || Path::new(&dylib).exists()
        || Path::new(&dll).exists();

    if lib_exists {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
        println!("cargo:warning=llama.cpp linked from {}", lib_dir);
    } else {
        println!("cargo:warning=llama.cpp not found at {}.", llama_build);
        println!("cargo:warning=  To enable FFI fallback for Qwen3.5/3.6 and exotic quants:");
        println!("cargo:warning=    1. ./scripts/build_llama_cpp.sh");
        println!("cargo:warning=    2. cd rust && cargo build --release --features llama-ffi");
        println!("cargo:warning=  Or set LLAMA_CPP_BUILD=/path/to/llama.cpp/build");
    }
}
