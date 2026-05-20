//! Manual FFI bindings for llama.cpp C API.
//! Only includes the minimal types and functions needed for inference.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::os::raw::c_float;

// ---------------------------------------------------------------------------
// Opaque structs
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_vocab {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_model {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _unused: [u8; 0],
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
pub type llama_pos = i32;
pub type llama_token = i32;
pub type llama_seq_id = i32;

pub type llama_progress_callback = Option<
    unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool,
>;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_vocab_type {
    NONE = 0,
    SPM = 1,
    BPE = 2,
    WPM = 3,
    UGM = 4,
    RWKV = 5,
    PLAMO2 = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_rope_scaling_type {
    UNSPECIFIED = -1,
    NONE = 0,
    LINEAR = 1,
    YARN = 2,
    LONGROPE = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_pooling_type {
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS = 2,
    LAST = 3,
    RANK = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_attention_type {
    UNSPECIFIED = -1,
    CAUSAL = 0,
    NON_CAUSAL = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_flash_attn_type {
    AUTO = -1,
    DISABLED = 0,
    ENABLED = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_split_mode {
    NONE = 0,
    LAYER = 1,
    ROW = 2,
    TENSOR = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_context_type {
    DEFAULT = 0,
    MTP = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum llama_model_kv_override_type {
    INT = 0,
    FLOAT = 1,
    BOOL = 2,
    STR = 3,
}

// ---------------------------------------------------------------------------
// llama_token_data
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: c_float,
    pub p: c_float,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

// ---------------------------------------------------------------------------
// llama_model_kv_override
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_model_kv_override {
    pub tag: llama_model_kv_override_type,
    pub key: [c_char; 128],
    pub value: llama_model_kv_override_value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union llama_model_kv_override_value {
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_bool: bool,
    pub val_str: [c_char; 128],
}

// ---------------------------------------------------------------------------
// llama_model_params  (sizeof = 72)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct llama_model_params {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

// ---------------------------------------------------------------------------
// llama_context_params  (sizeof = 144)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
}

// ---------------------------------------------------------------------------
// llama_batch  (sizeof = 56)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------
extern "C" {
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);

    pub fn llama_n_ctx_train(model: *const llama_model) -> i32;
    pub fn llama_n_embd(model: *const llama_model) -> i32;
    pub fn llama_n_layer(model: *const llama_model) -> i32;
    pub fn llama_n_vocab(vocab: *const llama_vocab) -> i32;
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;

    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;

    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);

    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;

    pub fn llama_get_logits(ctx: *mut llama_context) -> *mut c_float;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;
    pub fn llama_get_embeddings(ctx: *mut llama_context) -> *mut c_float;
    pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;

    // Sampler functions
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_sample(
        sampler: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_free(sampler: *mut llama_sampler);
}
