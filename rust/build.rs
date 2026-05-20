use std::env;

fn main() {
    // Path to llama.cpp build directory
    let llama_build = "/home/xander/Documents/llama.cpp/build";
    let llama_bin = format!("{}/bin", llama_build);

    println!("cargo:rustc-link-search=native={}", llama_bin);
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=llama-common");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    // Embed rpath so the binary finds libs at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_bin);

    // Tell cargo to rerun if the library changes
    println!("cargo:rerun-if-changed={}/libllama.so", llama_bin);
}
