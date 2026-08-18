//! LeafcutterLLM — Memory-safe LLM inference engine

pub mod api;
pub mod backend;
pub mod bridge;
pub mod cache;
pub mod config;
pub mod detect;

/// Deterministic-mode controls (kimi-k3-in-c "determinism contract"):
/// `LEAFCUTTER_DETERMINISTIC=1` forces serial, f64-accumulated dot products
/// and disables the numeric-regime switches (Q8_K integer dot, AVX2 dual-
/// accumulator FMA splits) so two runs on the same model produce
/// bit-identical logits regardless of machine or thread count.
pub mod deterministic;

/// Background safety monitor — observes CPU temp/RSS and prints warnings
/// to stderr.  Never throttles execution; pure advisory.
pub mod cpu_monitor;
pub mod gguf_provider;

pub mod init;

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

    /// Check whether the real llama.cpp backend is available.
    pub const fn is_available() -> bool { false }

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
pub mod ollama_backend;
pub mod ornith_config;
pub mod ornith_kernels;
pub mod streaming_ornith;
pub mod profiles;
pub mod bpe_tokenizer;
pub mod safetensor_backend;
pub mod safetensors_loader;
pub mod shard;
pub mod tokenizer;
pub mod launch;
