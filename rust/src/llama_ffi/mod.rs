//! Safe Rust wrapper around llama.cpp's C API via FFI.
//!
//! This module provides a minimal, safe interface to llama.cpp for inference.
//! It handles model loading, tokenization, forward passes, and logits extraction.

mod bindings;

use std::ffi::{c_char, CString};
use std::path::Path;
use std::ptr::NonNull;

pub use bindings::{
    llama_context, llama_context_params, llama_model, llama_model_params,
    llama_pos, llama_token, llama_vocab, llama_batch,
};

/// Check whether the real llama.cpp backend is available.
pub const fn is_available() -> bool { true }

/// Safe wrapper around a loaded llama.cpp model.
pub struct LlamaModel {
    ptr: NonNull<llama_model>,
}

/// Safe wrapper around a llama.cpp inference context.
pub struct LlamaContext {
    ptr: NonNull<llama_context>,
    model: NonNull<llama_model>,
    n_vocab: i32,
    n_embd: i32,
}

/// Opaque batch handle for feeding tokens to llama.cpp.
pub struct LlamaBatch {
    inner: llama_batch,
    capacity: i32,
}

// SAFETY: LlamaModel and LlamaContext wrap llama.cpp opaque pointers.
// These are **not** thread-safe at the C level — concurrent
// access to a single Context can corrupt the KV cache. The Native
// backend wraps a per-context mutex (NativeStreamingEngine + HybridEngine).
// Callers in `LeafcutterEngine::generate` MUST ensure each context is
// owned by one request at a time. Direct use of these raw unsafe impls
// from multiple threads without synchronisation is a data race.
//
// TODO: replace these blanket impls with `Mutex<LlamaContext>` (the
// Native backend already does this) so the unsafe impls can be removed
// once all FFI call sites are audited.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}
unsafe impl Send for LlamaContext {}
unsafe impl Sync for LlamaContext {}

impl LlamaModel {
    /// Load a GGUF model from disk.
    pub fn load(path: &Path, n_gpu_layers: i32) -> Result<Self, String> {
        let c_path = CString::new(path.to_str().ok_or("Invalid path")?)
            .map_err(|e| format!("CString error: {}", e))?;

        unsafe {
            let mut mparams = bindings::llama_model_default_params();
            mparams.n_gpu_layers = n_gpu_layers;

            let ptr = bindings::llama_model_load_from_file(c_path.as_ptr(), mparams);
            if ptr.is_null() {
                return Err(format!("Failed to load model from {:?}", path));
            }
            Ok(Self {
                ptr: NonNull::new_unchecked(ptr),
            })
        }
    }

    pub fn n_vocab(&self) -> i32 {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.ptr.as_ptr());
            bindings::llama_vocab_n_tokens(vocab)
        }
    }

    pub fn n_embd(&self) -> i32 {
        unsafe { bindings::llama_n_embd(self.ptr.as_ptr()) }
    }

    pub fn n_ctx_train(&self) -> i32 {
        unsafe { bindings::llama_n_ctx_train(self.ptr.as_ptr()) }
    }

    pub fn n_layer(&self) -> i32 {
        unsafe { bindings::llama_n_layer(self.ptr.as_ptr()) }
    }

    pub fn vocab(&self) -> *const llama_vocab {
        unsafe { bindings::llama_model_get_vocab(self.ptr.as_ptr()) }
    }

    /// Returns true if the model should add BOS token.
    pub fn add_bos_token(&self) -> bool {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.ptr.as_ptr());
            bindings::llama_vocab_get_add_bos(vocab)
        }
    }

    pub fn bos_token(&self) -> llama_token {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.ptr.as_ptr());
            bindings::llama_vocab_bos(vocab)
        }
    }

    pub fn eos_token(&self) -> llama_token {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.ptr.as_ptr());
            bindings::llama_vocab_eos(vocab)
        }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            bindings::llama_model_free(self.ptr.as_ptr());
        }
    }
}

impl LlamaContext {
    /// Create a new inference context from a loaded model.
    pub fn new(model: &LlamaModel, n_ctx: u32, n_threads: i32) -> Result<Self, String> {
        unsafe {
            let mut cparams = bindings::llama_context_default_params();
            cparams.n_ctx = n_ctx;
            cparams.n_batch = n_ctx;
            cparams.n_ubatch = n_ctx;
            cparams.n_threads = n_threads;
            cparams.n_threads_batch = n_threads;

            let ptr = bindings::llama_init_from_model(model.ptr.as_ptr(), cparams);
            if ptr.is_null() {
                return Err("Failed to create llama context".to_string());
            }
            Ok(Self {
                ptr: NonNull::new_unchecked(ptr),
                model: model.ptr,
                n_vocab: model.n_vocab(),
                n_embd: model.n_embd(),
            })
        }
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<llama_token> {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.model.as_ptr());
            // Strip embedded NUL bytes — CString::new fails on them, and
            // they would silently truncate the input.  NULs in user text
            // are almost never intentional.
            let clean: String = text.replace('\0', "");
            let c_text = CString::new(clean).expect("NULs stripped; empty string is valid CString");
            let text_len = c_text.as_bytes().len() as i32;

            // First call: get required buffer size
            // llama_tokenize returns -n_tokens when buffer is too small (including when n_tokens_max == 0)
            let n_needed = bindings::llama_tokenize(
                vocab,
                c_text.as_ptr(),
                text_len,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            );
            if n_needed == 0 {
                return Vec::new();
            }
            let n_needed = n_needed.abs();

            // Second call: fill buffer
            let mut tokens = vec![0i32; n_needed as usize];
            let n_written = bindings::llama_tokenize(
                vocab,
                c_text.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                n_needed,
                add_special,
                parse_special,
            );
            tokens.truncate(n_written.max(0) as usize);
            tokens
        }
    }

    /// Convert a token ID back to its text representation.
    pub fn token_to_piece(&self, token: llama_token) -> String {
        unsafe {
            let vocab = bindings::llama_model_get_vocab(self.model.as_ptr());
            let mut buf = vec![0u8; 256];
            let len = bindings::llama_token_to_piece(
                vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                true,
            );
            if len > 0 {
                buf.truncate(len as usize);
                String::from_utf8_lossy(&buf).into_owned()
            } else {
                String::new()
            }
        }
    }

    /// Run a forward pass for the given tokens and return logits for the last token.
    pub fn forward(&mut self, tokens: &[llama_token]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("Empty token list".to_string());
        }

        let mut batch = LlamaBatch::new(tokens.len() as i32, 1)?;
        batch.set_tokens(tokens);

        unsafe {
            let ret = bindings::llama_decode(self.ptr.as_ptr(), batch.inner);
            if ret != 0 {
                return Err(format!("llama_decode failed with code {}", ret));
            }

            // Get logits for the last token
            let logits_ptr = bindings::llama_get_logits_ith(self.ptr.as_ptr(), tokens.len() as i32 - 1);
            if logits_ptr.is_null() {
                return Err("llama_get_logits_ith returned null".to_string());
            }

            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab as usize);
            Ok(logits.to_vec())
        }
    }

    /// Decode a single token at a specific position (for autoregressive generation).
    pub fn decode_single(&mut self, token: llama_token, pos: llama_pos) -> Result<(), String> {
        let mut batch = LlamaBatch::new(1, 1)?;
        batch.set_single_token(token, pos);

        unsafe {
            let ret = bindings::llama_decode(self.ptr.as_ptr(), batch.inner);
            if ret != 0 {
                return Err(format!("llama_decode failed with code {}", ret));
            }
        }
        Ok(())
    }

    /// Sample the next token using greedy (argmax) selection.
    pub fn sample_greedy(&self) -> llama_token {
        unsafe {
            let logits_ptr = bindings::llama_get_logits(self.ptr.as_ptr());
            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab as usize);
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as llama_token)
                .unwrap_or(0)
        }
    }

    /// Sample the next token with temperature.
    /// temperature = 0.0 → greedy. temperature > 1.0 → more random.
    pub fn sample_temperature(&self, temperature: f32) -> llama_token {
        if temperature <= 0.0 {
            return self.sample_greedy();
        }

        unsafe {
            let logits_ptr = bindings::llama_get_logits(self.ptr.as_ptr());
            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab as usize);

            // Softmax with temperature
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= sum;
            }

            // Categorical sampling
            let r: f32 = rand::random();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    return i as llama_token;
                }
            }
            (probs.len() - 1) as llama_token
        }
    }

    /// Generate text autoregressively from a prompt.
    pub fn generate(
        &mut self,
        prompt_tokens: &[llama_token],
        max_tokens: usize,
        temperature: f32,
        eos_token: llama_token,
    ) -> Vec<llama_token> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }

        let mut generated = Vec::new();

        // ---- Prefill: feed all prompt tokens at once ----
        {
            let mut batch = LlamaBatch::new(prompt_tokens.len() as i32, 1)
                .expect("batch alloc failed");
            batch.set_tokens(prompt_tokens);

            unsafe {
                let ret = bindings::llama_decode(self.ptr.as_ptr(), batch.inner);
                if ret != 0 {
                    eprintln!("Prefill decode failed: {}", ret);
                    return Vec::new();
                }
            }
        }

        let mut pos = prompt_tokens.len() as llama_pos;

        // Sample first token from last prompt position
        let mut next_token = if temperature <= 0.0 {
            self.sample_greedy()
        } else {
            self.sample_temperature(temperature)
        };

        // ---- Autoregressive loop ----
        for _ in 0..max_tokens {
            if next_token == eos_token {
                break;
            }
            generated.push(next_token);

            // Decode the newly generated token
            if let Err(e) = self.decode_single(next_token, pos) {
                eprintln!("Decode error at pos {}: {}", pos, e);
                break;
            }
            pos += 1;

            next_token = if temperature <= 0.0 {
                self.sample_greedy()
            } else {
                self.sample_temperature(temperature)
            };
        }

        generated
    }

    /// Get embeddings (final hidden states) for all tokens in the last forward pass.
    pub fn get_embeddings(&mut self, n_tokens: usize) -> Result<Vec<f32>, String> {
        unsafe {
            let emb_ptr = bindings::llama_get_embeddings(self.ptr.as_ptr());
            if emb_ptr.is_null() {
                return Err("llama_get_embeddings returned null".to_string());
            }
            let emb = std::slice::from_raw_parts(emb_ptr, n_tokens * self.n_embd as usize);
            Ok(emb.to_vec())
        }
    }

    pub fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    pub fn n_embd(&self) -> i32 {
        self.n_embd
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            bindings::llama_free(self.ptr.as_ptr());
        }
    }
}

impl LlamaBatch {
    /// Allocate a new batch with the given capacity.
    pub fn new(n_tokens: i32, n_seq_max: i32) -> Result<Self, String> {
        unsafe {
            let inner = bindings::llama_batch_init(n_tokens, 0, n_seq_max);
            if inner.token.is_null() {
                return Err("llama_batch_init failed".to_string());
            }
            Ok(Self { inner, capacity: n_tokens })
        }
    }

    /// Fill the batch with tokens for a single sequence.
    pub fn set_tokens(&mut self, tokens: &[llama_token]) {
        assert!(
            tokens.len() <= self.capacity as usize,
            "Batch overflow: {} > {}",
            tokens.len(),
            self.capacity
        );

        unsafe {
            for (i, &tok) in tokens.iter().enumerate() {
                *self.inner.token.add(i) = tok;
                *self.inner.pos.add(i) = i as llama_pos;
                *self.inner.n_seq_id.add(i) = 1;
                *(*self.inner.seq_id.add(i)).add(0) = 0;
                *self.inner.logits.add(i) = if i == tokens.len() - 1 { 1 } else { 0 };
            }
            self.inner.n_tokens = tokens.len() as i32;
        }
    }

    /// Set a single token at a specific position (for decode step).
    pub fn set_single_token(&mut self, token: llama_token, pos: llama_pos) {
        unsafe {
            *self.inner.token = token;
            *self.inner.pos = pos;
            *self.inner.n_seq_id = 1;
            *(*self.inner.seq_id).add(0) = 0;
            *self.inner.logits = 1;
            self.inner.n_tokens = 1;
        }
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe {
            bindings::llama_batch_free(self.inner);
        }
    }
}

/// Initialize the llama.cpp backend (call once at program startup).
pub fn backend_init() {
    unsafe {
        bindings::llama_backend_init();
    }
}

/// Free the llama.cpp backend (call once at program shutdown).
pub fn backend_free() {
    unsafe {
        bindings::llama_backend_free();
    }
}
