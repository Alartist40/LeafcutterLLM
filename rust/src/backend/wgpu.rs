//! WGPU compute backend — GPU-accelerated matmul via compute shaders
//!
//! Uses the system's default GPU (Vulkan, Metal, DX12, or WebGPU).
//! Currently implements matmul on GPU; all other ops fall back to CpuBackend.
//!
//! Usage:
//!   let backend = WgpuBackend::new().expect("GPU required");
//!   Tensor::set_global_backend(backend);

use super::cpu::CpuBackend;
use super::Backend;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // m, k, n

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let m = dims.x;
    let k = dims.y;
    let n = dims.z;

    if (row >= m || col >= n) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        sum = sum + a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}
"#;

/// GPU-backed compute backend.
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    matmul_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    /// Initialize WGPU and compile compute shaders.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("No GPU adapter found: {}", e))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Leafcutter WGPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|e| format!("Failed to create GPU device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MATMUL_WGSL)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            bind_group_layout,
        })
    }
}

impl Backend for WgpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // Small matrices are faster on CPU due to GPU kernel launch overhead
        if m * k * n < 256 * 256 * 256 {
            return CpuBackend.matmul(a, b, m, k, n);
        }

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims = [m as u32, k as u32, n as u32];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("matmul_encoder") });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                ((n + 7) / 8) as u32,
                ((m + 7) / 8) as u32,
                1,
            );
        }

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, (m * n * std::mem::size_of::<f32>()) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    // All other ops fall back to CPU — GPU kernels for element-wise ops
    // don't beat CPU due to upload/download overhead.
    fn vec_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.vec_add(a, b) }
    fn vec_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.vec_mul(a, b) }
    fn vec_scale(&self, a: &[f32], scale: f32) -> Vec<f32> { CpuBackend.vec_scale(a, scale) }
    fn vec_scale_mul(&self, a: &[f32], scale: f32, b: &[f32]) -> Vec<f32> { CpuBackend.vec_scale_mul(a, scale, b) }
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> { CpuBackend.rms_norm(x, weight, eps, hidden_size) }
    fn silu(&self, x: &[f32]) -> Vec<f32> { CpuBackend.silu(x) }
    fn softmax(&self, x: &[f32], hidden_size: usize) -> Vec<f32> { CpuBackend.softmax(x, hidden_size) }
    fn sum_sq(&self, x: &[f32]) -> f32 { CpuBackend.sum_sq(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires GPU"]
    fn test_wgpu_matmul() {
        let backend = WgpuBackend::new().expect("GPU required for this test");

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let c = backend.matmul(&a, &b, 2, 2, 2);

        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[ignore = "Requires GPU"]
    fn test_wgpu_matmul_large() {
        let backend = WgpuBackend::new().expect("GPU required for this test");

        let m = 256;
        let k = 256;
        let n = 256;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.001).collect();

        let c_gpu = backend.matmul(&a, &b, m, k, n);
        let c_cpu = CpuBackend.matmul(&a, &b, m, k, n);

        for i in 0..c_gpu.len() {
            let rel_err = (c_gpu[i] - c_cpu[i]).abs() / c_cpu[i].abs().max(1.0);
            assert!(rel_err < 1e-4,
                "GPU/CPU mismatch at {}: gpu={}, cpu={}, rel_err={}", i, c_gpu[i], c_cpu[i], rel_err);
        }
    }
}
