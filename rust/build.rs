fn main() {
    // Allow overriding the llama.cpp build path via env var.
    let llama_build = std::env::var("LLAMA_CPP_BUILD")
        .unwrap_or_else(|_| "/home/xander/Documents/llama.cpp/build".to_string());

    let lib_exists = std::path::Path::new(&format!("{}/bin/libllama.so", llama_build)).exists()
        || std::path::Path::new(&format!("{}/bin/libllama.dylib", llama_build)).exists()
        || std::path::Path::new(&format!("{}/bin/llama.dll", llama_build)).exists();

    if lib_exists {
        println!("cargo:rustc-link-search=native={}/bin", llama_build);
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/bin", llama_build);
    } else {
        println!("cargo:warning=llama.cpp not found at {}. FFI disabled. Build with LLAMA_CPP_BUILD=/path/to/build --features llama-ffi to enable.", llama_build);
    }
}
