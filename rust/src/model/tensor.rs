//! f32 tensor implementation with pluggable compute backend

use crate::backend::{default_backend, set_global_backend, Backend};
use crate::kernels::iq4_nl::Matrix as IQ4NLMatrix;
use crate::kernels::q4_0::Matrix as Q4Matrix;
use crate::kernels::q4_k::Matrix as Q4KMatrix;
use crate::kernels::q5_k::Matrix as Q5KMatrix;
use crate::kernels::q6_k::Matrix as Q6KMatrix;
use crate::kernels::q8_0::Matrix as Q8Matrix;

/// Native quantized weight data attached to a Tensor.
/// When present, matmul dispatches to a format-specific kernel
/// for memory-bandwidth savings. All other ops use `data` (f32).
#[derive(Clone)]
#[allow(non_camel_case_types)]
pub enum QuantizedData {
    IQ4_NL(IQ4NLMatrix),
    Q4_0(Q4Matrix),
    Q4_K(Q4KMatrix),
    Q5_K(Q5KMatrix),
    Q6_K(Q6KMatrix),
    Q8_0(Q8Matrix),
}

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// Optional quantized weights for fast matmul.
    q_data: Option<QuantizedData>,
    backend: &'static dyn Backend,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
            .field("quantized", &self.q_data.is_some())
            .field("backend", &"<dyn Backend>")
            .finish()
    }
}

impl Tensor {
    /// Create a tensor filled with zeros, using the global backend.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
            q_data: None,
            backend: default_backend(),
        }
    }

    /// Create a tensor from raw data, using the global backend.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            panic!("Tensor::from_vec: data.len={} != shape={:?} (expected={})", data.len(), shape, expected);
        }
        Self {
            shape,
            data,
            q_data: None,
            backend: default_backend(),
        }
    }

    /// Create a tensor with a specific backend.
    pub fn from_vec_with_backend(data: Vec<f32>, shape: Vec<usize>, backend: &'static dyn Backend) -> Self {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            panic!("Tensor::from_vec_with_backend: data.len={} != shape={:?} (expected={})", data.len(), shape, expected);
        }
        Self { shape, data, q_data: None, backend }
    }

    /// Create a tensor from Q8_0 quantized weights.
    /// Stores both Q8_0 (for fast INT8 matmul) and f32 (for other ops).
    pub fn from_q8_0(q8: Q8Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q8.rows * q8.cols, shape.iter().product::<usize>());
        let data = q8.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q8_0(q8)),
            backend: default_backend(),
        }
    }

    /// Create a tensor from Q8_0 quantized weights WITHOUT f32 copy.
    /// Use for weight tensors that are only consumed by matmul.
    pub fn from_q8_0_only(q8: Q8Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q8.rows * q8.cols, shape.iter().product::<usize>());
        Self {
            shape,
            data: Vec::new(),
            q_data: Some(QuantizedData::Q8_0(q8)),
            backend: default_backend(),
        }
    }

    /// Create a tensor from Q4_0 quantized weights.
    pub fn from_q4_0(q4: Q4Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        let data = q4.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q4_0(q4)),
            backend: default_backend(),
        }
    }

    pub fn from_q4_0_only(q4: Q4Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        Self { shape, data: Vec::new(), q_data: Some(QuantizedData::Q4_0(q4)), backend: default_backend() }
    }

    /// Create a tensor from Q4_K quantized weights.
    pub fn from_q4_k(q4: Q4KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        let data = q4.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q4_K(q4)),
            backend: default_backend(),
        }
    }

    pub fn from_q4_k_only(q4: Q4KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        Self { shape, data: Vec::new(), q_data: Some(QuantizedData::Q4_K(q4)), backend: default_backend() }
    }

    /// Create a tensor from IQ4_NL quantized weights.
    pub fn from_iq4_nl(q4: IQ4NLMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        let data = q4.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::IQ4_NL(q4)),
            backend: default_backend(),
        }
    }

    pub fn from_iq4_nl_only(q4: IQ4NLMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        Self { shape, data: Vec::new(), q_data: Some(QuantizedData::IQ4_NL(q4)), backend: default_backend() }
    }

    /// Create a tensor from Q5_K quantized weights.
    pub fn from_q5_k(q5: Q5KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q5.rows * q5.cols, shape.iter().product::<usize>());
        let data = q5.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q5_K(q5)),
            backend: default_backend(),
        }
    }

    pub fn from_q5_k_only(q5: Q5KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q5.rows * q5.cols, shape.iter().product::<usize>());
        Self { shape, data: Vec::new(), q_data: Some(QuantizedData::Q5_K(q5)), backend: default_backend() }
    }

    /// Create a tensor from Q6_K quantized weights.
    pub fn from_q6_k(q6: Q6KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q6.rows * q6.cols, shape.iter().product::<usize>());
        let data = q6.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q6_K(q6)),
            backend: default_backend(),
        }
    }

    pub fn from_q6_k_only(q6: Q6KMatrix, shape: Vec<usize>) -> Self {
        assert_eq!(q6.rows * q6.cols, shape.iter().product::<usize>());
        Self { shape, data: Vec::new(), q_data: Some(QuantizedData::Q6_K(q6)), backend: default_backend() }
    }

    /// Returns true if this tensor has native quantized data for fast matmul.
    pub fn is_quantized(&self) -> bool {
        self.q_data.is_some()
    }

    /// Materialize f32 data from quantized weights if not already present.
    /// After this call, `self.data` is populated and `q_data` is retained
    /// for fast quantized matmul.  No-op if `data` is already non-empty.
    pub fn materialize_data(&mut self) {
        if !self.data.is_empty() {
            return;
        }
        let new_data: Vec<f32> = match self.q_data.as_ref() {
            Some(QuantizedData::Q8_0(m)) => m.dequantize(),
            Some(QuantizedData::Q4_0(m)) => m.dequantize(),
            Some(QuantizedData::Q4_K(m)) => m.dequantize(),
            Some(QuantizedData::Q5_K(m)) => m.dequantize(),
            Some(QuantizedData::Q6_K(m)) => m.dequantize(),
            Some(QuantizedData::IQ4_NL(m)) => m.dequantize(),
            None => return,
        };
        self.data = new_data;
    }

    /// Attach Q8_0 quantized data to an existing tensor (used by shard loader
    /// after it has separately dequantized + transposed the f32 data).
    pub fn with_q8_0(mut self, q8: crate::kernels::q8_0::Matrix) -> Self {
        self.q_data = Some(QuantizedData::Q8_0(q8));
        self
    }

    /// Attach Q4_0 quantized data to an existing tensor.
    pub fn with_q4_0(mut self, q4: crate::kernels::q4_0::Matrix) -> Self {
        self.q_data = Some(QuantizedData::Q4_0(q4));
        self
    }

    /// Report approximate memory used by quantized data (bytes).
    pub fn quantized_memory_bytes(&self) -> usize {
        match &self.q_data {
            Some(QuantizedData::Q4_K(q4)) => q4.blocks.len() * 144,
            Some(QuantizedData::Q5_K(q5)) => q5.blocks.len() * 176,
            Some(QuantizedData::Q6_K(q6)) => q6.blocks.len() * 210,
            Some(QuantizedData::Q8_0(q8)) => q8.blocks.len() * 34,
            Some(QuantizedData::Q4_0(q4)) => q4.blocks.len() * 18,
            Some(QuantizedData::IQ4_NL(q4)) => q4.blocks.len() * 18,
            None => 0,
        }
    }

    /// Total resident bytes of this tensor: quantized blocks plus any
    /// materialized f32 data (4 bytes each).  This is the real memory the
    /// tensor occupies in the layer cache — cache accounting must use it,
    /// not `quantized_memory_bytes` alone, or materialized tensors silently
    /// exceed the budget and blow through available RAM.
    pub fn resident_bytes(&self) -> usize {
        self.quantized_memory_bytes()
            .saturating_add(self.data.len().saturating_mul(4))
    }

    /// Slice one expert out of a 3-D expert tensor (GGUF dims `[d0, d1, d2]`
    /// where `d2` = number of experts).
    ///
    /// GGUF / llama.cpp store these row-major over `[expert, d1, d0]` with
    /// `d0` contiguous and the expert axis outermost (see
    /// `llama-model.cpp` `create_tensor_gate_up_exps`), so expert `e`
    /// occupies elements `[e*d1*d0, (e+1)*d1*d0)` — rows `[e*d1, (e+1)*d1)`
    /// of the loader's collapsed `[d1*d2, d0]` matrix.
    ///
    /// Returns a 2-D `[d0, d1]` tensor for that expert — quantized (no f32
    /// materialization) when the parent carries `q_data`, otherwise a plain
    /// f32 slice.  Returns `None` for non-3-D tensors or when the parent has
    /// neither quantized nor f32 data.
    pub fn expert_slice(&self, expert_idx: usize) -> Option<Tensor> {
        if self.shape.len() != 3 {
            return None;
        }
        let d0 = self.shape[0];
        let d1 = self.shape[1];
        let d2 = self.shape[2];
        if expert_idx >= d2 {
            return None;
        }
        let shape = vec![d0, d1];
        if let Some(q) = &self.q_data {
            let sub = match q {
                QuantizedData::Q4_K(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::Q4_K(Q4KMatrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
                QuantizedData::Q5_K(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::Q5_K(Q5KMatrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
                QuantizedData::Q6_K(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::Q6_K(Q6KMatrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
                QuantizedData::Q8_0(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::Q8_0(Q8Matrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
                QuantizedData::Q4_0(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::Q4_0(Q4Matrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
                QuantizedData::IQ4_NL(m) => {
                    let bpr = m.blocks_per_row();
                    let start = expert_idx * d1 * bpr;
                    QuantizedData::IQ4_NL(IQ4NLMatrix {
                        rows: d1,
                        cols: d0,
                        blocks: m.blocks[start..start + d1 * bpr].to_vec(),
                    })
                }
            };
            Some(Tensor { shape, data: Vec::new(), q_data: Some(sub), backend: self.backend })
        } else {
            if self.data.is_empty() {
                return None;
            }
            // `.data` is flat in `[expert, d1, d0]` order; the returned
            // `[d0, d1]` tensor must be the transposed layout (element
            // `(o, i) = W[i, o]`) so the standard f32 matmul produces the
            // same result as the quantized B^T GEMM on the sliced blocks.
            let start = expert_idx * d1 * d0;
            let mut sub = Vec::with_capacity(d0 * d1);
            for o in 0..d0 {
                for i in 0..d1 {
                    sub.push(self.data[start + i * d0 + o]);
                }
            }
            Some(Tensor::from_vec(sub, shape))
        }
    }

    /// Set the global backend for all new Tensors.
    pub fn set_global_backend(backend: &'static dyn Backend) {
        set_global_backend(backend);
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Matrix multiplication: self @ other
    /// self: [m, k], other: [k, n], result: [m, n]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        assert_eq!(k, other.shape[0]);

        let profile = std::env::var("LEAFCUTTER_PROFILE").is_ok();
        let t0 = if profile { Some(std::time::Instant::now()) } else { None };

        // Fast path: if other has quantized weights, use native GEMM.
        // Quantized matrices are stored in native GGUF layout [n, k];
        // we compute C = A @ B^T using transposed-B kernels.
        if let Some(ref q) = other.q_data {
            let mut result = vec![0.0f32; m * n];
            let qtype = match q {
                QuantizedData::Q8_0(_) => "Q8_0",
                QuantizedData::Q4_0(_) => "Q4_0",
                QuantizedData::Q4_K(_) => "Q4_K",
                QuantizedData::IQ4_NL(_) => "IQ4_NL",
                QuantizedData::Q5_K(_) => "Q5_K",
                QuantizedData::Q6_K(_) => "Q6_K",
            };
            match q {
                QuantizedData::Q8_0(q8) => {
                    assert_eq!(q8.cols, k, "Q8_0 cols mismatch");
                    assert_eq!(q8.rows, n, "Q8_0 rows mismatch");
                    crate::kernels::int8_gemm::q8_0_matmul_transposed_b(&self.data, q8, &mut result, m, k, n);
                }
                QuantizedData::Q4_0(q4) => {
                    assert_eq!(q4.cols, k, "Q4_0 cols mismatch");
                    assert_eq!(q4.rows, n, "Q4_0 rows mismatch");
                    crate::kernels::int8_gemm::q4_0_matmul_transposed_b(&self.data, q4, &mut result, m, k, n);
                }
                QuantizedData::Q4_K(q4) => {
                    assert_eq!(q4.cols, k, "Q4_K cols mismatch");
                    assert_eq!(q4.rows, n, "Q4_K rows mismatch");
                    crate::kernels::q4_k_gemm::q4_k_matmul_transposed_b(&self.data, q4, &mut result, m, k, n);
                }
                QuantizedData::IQ4_NL(q4) => {
                    assert_eq!(q4.cols, k, "IQ4_NL cols mismatch");
                    assert_eq!(q4.rows, n, "IQ4_NL rows mismatch");
                    crate::kernels::iq4_nl_gemm::iq4_nl_matmul_transposed_b(&self.data, q4, &mut result, m, k, n);
                }
                QuantizedData::Q5_K(q5) => {
                    assert_eq!(q5.cols, k, "Q5_K cols mismatch");
                    assert_eq!(q5.rows, n, "Q5_K rows mismatch");
                    crate::kernels::q5_k_gemm::q5_k_matmul_transposed_b(&self.data, q5, &mut result, m, k, n);
                }
                QuantizedData::Q6_K(q6) => {
                    assert_eq!(q6.cols, k, "Q6_K cols mismatch");
                    assert_eq!(q6.rows, n, "Q6_K rows mismatch");
                    crate::kernels::q6_k_gemm::q6_k_matmul_transposed_b(&self.data, q6, &mut result, m, k, n);
                }
            }
            if let Some(t0) = t0 {
                let elapsed = t0.elapsed();
                eprintln!("[PROFILE] matmul {:>6}  m={:>4} k={:>4} n={:>6}  {:>8.2}ms",
                    qtype, m, k, n, elapsed.as_secs_f32() * 1000.0);
            }
            return Self::from_vec_with_backend(result, vec![m, n], self.backend);
        }

        let result = self.backend.matmul(&self.data, &other.data, m, k, n);
        if let Some(t0) = t0 {
            let elapsed = t0.elapsed();
            eprintln!("[PROFILE] matmul {:>6}  m={:>4} k={:>4} n={:>6}  {:>8.2}ms",
                "f32", m, k, n, elapsed.as_secs_f32() * 1000.0);
        }
        Self::from_vec_with_backend(result, vec![m, n], self.backend)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.size(), other.size());
        let data = self.backend.vec_add(&self.data, &other.data);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// RMSNorm: x * rsqrt(mean(x^2) + epsilon) * weight
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let hidden_size = self.shape.last().copied().unwrap_or(1);
        let data = self.backend.rms_norm(&self.data, &weight.data, eps, hidden_size);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// RMSNorm with weight offset: x * rsqrt(mean(x^2) + epsilon) * (weight + offset)
    pub fn rms_norm_with_offset(&self, weight: &Tensor, eps: f32, weight_offset: f32) -> Tensor {
        let hidden_size = self.shape.last().copied().unwrap_or(1);
        let data = self.backend.rms_norm_with_offset(&self.data, &weight.data, eps, hidden_size, weight_offset);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// SiLU activation: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let data = self.backend.silu(&self.data);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// Softmax over last dimension
    pub fn softmax_last_dim(&self) -> Tensor {
        let hidden_size = self.shape.last().copied().unwrap_or(1);
        let data = self.backend.softmax(&self.data, hidden_size);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// Reshape to new shape (total size must match)
    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        assert_eq!(self.size(), shape.iter().product::<usize>());
        Self {
            shape,
            data: self.data.clone(),
            q_data: None, // reshape invalidates quantized block layout
            backend: self.backend,
        }
    }

    /// Transpose a 2D tensor: [m, n] -> [n, m]
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }
        // Transpose invalidates quantized block layout
        Self::from_vec_with_backend(result, vec![n, m], self.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rms_norm() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let w = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
        let y = x.rms_norm(&w, 1e-5);
        assert!(y.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = x.softmax_last_dim();
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
