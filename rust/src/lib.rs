//! LeafcutterLLM — Memory-safe LLM inference engine

#[cfg(feature = "llama-ffi")]
pub mod api;
pub mod backend;
pub mod bridge;
pub mod cache;

pub mod inference;
pub mod kernels;

#[cfg(feature = "llama-ffi")]
pub mod llama_ffi;

#[cfg(not(feature = "llama-ffi"))]
pub mod llama_ffi {
    //! Stub module when llama.cpp FFI is not available.
    pub fn backend_init() {}
    pub fn backend_free() {}
    pub struct LlamaModel;
    pub struct LlamaContext;
    pub struct LlamaBatch;
    impl LlamaModel {
        pub fn load(_path: &std::path::Path, _n_gpu_layers: i32) -> Result<Self, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn n_vocab(&self) -> i32 { 0 }
        pub fn n_embd(&self) -> i32 { 0 }
        pub fn n_layer(&self) -> i32 { 0 }
        pub fn n_ctx_train(&self) -> i32 { 0 }
        pub fn add_bos_token(&self) -> bool { false }
        pub fn bos_token(&self) -> i32 { 0 }
        pub fn eos_token(&self) -> i32 { 2 }
    }
    impl LlamaContext {
        pub fn new(_model: &LlamaModel, _n_ctx: u32, _n_threads: i32) -> Result<Self, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn tokenize(&self, _text: &str, _add_special: bool, _parse_special: bool) -> Vec<i32> {
            vec![]
        }
        pub fn token_to_piece(&self, _token: i32) -> String {
            String::new()
        }
        pub fn forward(&mut self, _tokens: &[i32]) -> Result<Vec<f32>, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn generate(&mut self, _prompt: &[i32], _max_tokens: usize, _temperature: f32, _eos_token: i32) -> Vec<i32> {
            vec![]
        }
        pub fn sample_greedy(&self) -> i32 { 0 }
        pub fn sample_temperature(&self, _t: f32) -> i32 { 0 }
    }
}

pub mod model;
pub mod shard;
pub mod tokenizer;
