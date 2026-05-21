fn main() {
    // Allow overriding the llama.cpp build path via env var.
    // On Pi: export LLAMA_CPP_BUILD=/home/pi/llama.cpp/build
    let llama_build = std::env::var("LLAMA_CPP_BUILD")
        .unwrap_or_else(|_| "/home/xander/Documents/llama.cpp/build".to_string());

    println!("cargo:rustc-link-search=native={}/bin", llama_build);
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/bin", llama_build);
}
