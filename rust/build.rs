use std::path::Path;

fn main() {
    // Allow overriding the llama.cpp build path via env var.
    let llama_build = std::env::var("LLAMA_CPP_BUILD")
        .unwrap_or_else(|_| {
            // Check submodule path first
            let submodule_build = Path::new("llama.cpp/build");
            if submodule_build.exists() {
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
        println!("cargo:warning=    1. git submodule update --init --recursive");
        println!("cargo:warning=    2. cd rust/llama.cpp && mkdir build && cd build");
        println!("cargo:warning=    3. cmake .. -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF");
        println!("cargo:warning=    4. cmake --build . --parallel $(nproc)");
        println!("cargo:warning=    5. cd ../.. && cargo build --release --features llama-ffi");
        println!("cargo:warning=  Or set LLAMA_CPP_BUILD=/path/to/llama.cpp/build");
    }
}
