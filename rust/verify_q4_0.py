#!/usr/bin/env python3
"""Parse Q4_0 tensor directly from GGUF and compare with Rust dequantization."""

import struct
import numpy as np

GGUF_PATH = "../models/Qwen3.5-0.8B-Q4_0.gguf"
TENSOR_NAME = b"blk.0.attn_qkv.weight"
EXPECTED_SHAPE = (1024, 6144)  # GGUF layout
BLOCK_SIZE = 32
GROUP_SIZE = 18  # 2 bytes scale + 16 bytes for 32 nibbles

# Simple GGUF parser to find tensor offset
with open(GGUF_PATH, "rb") as f:
    magic = f.read(4)
    print(f"Magic: {magic}")
    version = struct.unpack("<I", f.read(4))[0]
    print(f"Version: {version}")
    
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Tensors: {n_tensors}, KV pairs: {n_kv}")
    
    # Skip metadata
    for _ in range(n_kv):
        key_len = struct.unpack("<Q", f.read(8))[0]
        key = f.read(key_len)
        val_type = struct.unpack("<I", f.read(4))[0]
        
        # Read value based on type
        if val_type == 0:  # UINT8
            f.read(1)
        elif val_type == 1:  # INT8
            f.read(1)
        elif val_type == 2:  # UINT16
            f.read(2)
        elif val_type == 3:  # INT16
            f.read(2)
        elif val_type == 4:  # UINT32
            f.read(4)
        elif val_type == 5:  # INT32
            f.read(4)
        elif val_type == 6:  # FLOAT32
            f.read(4)
        elif val_type == 7:  # BOOL
            f.read(1)
        elif val_type == 8:  # STRING
            s_len = struct.unpack("<Q", f.read(8))[0]
            f.read(s_len)
        elif val_type == 9:  # ARRAY
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            # Rough size estimate
            type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:8, 10:8}
            if arr_type in type_sizes:
                f.read(type_sizes[arr_type] * arr_len)
            else:
                raise ValueError(f"Unknown array type: {arr_type}")
        elif val_type == 10:  # UINT64
            f.read(8)
        elif val_type == 11:  # INT64
            f.read(8)
        elif val_type == 12:  # FLOAT64
            f.read(8)
        else:
            raise ValueError(f"Unknown value type: {val_type}")
    
    # Now read tensor info
    tensor_offset = None
    tensor_type = None
    for _ in range(n_tensors):
        name_len = struct.unpack("<Q", f.read(8))[0]
        name = f.read(name_len)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q"*n_dims, f.read(8*n_dims))
        typ = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        
        if name == TENSOR_NAME:
            tensor_offset = offset
            tensor_type = typ
            print(f"Found {name.decode()}: dims={dims}, type={typ}, offset={offset}")
    
    if tensor_offset is None:
        raise ValueError("Tensor not found")
    
    # Go to tensor data
    # In GGUF v3, tensor data starts at the end of the header
    # The offset is relative to the start of the tensor data section
    # We need to find where the tensor data section starts
    # After tensor info, there's padding to alignment
    current_pos = f.tell()
    alignment = 32  # Common alignment
    padding = (alignment - current_pos % alignment) % alignment
    tensor_data_start = current_pos + padding
    
    f.seek(tensor_data_start + tensor_offset)
    
    total_elements = np.prod(EXPECTED_SHAPE)
    num_blocks = total_elements // BLOCK_SIZE
    raw_bytes = f.read(num_blocks * GROUP_SIZE)
    print(f"Read {len(raw_bytes)} bytes for {num_blocks} blocks")

# Dequantize Q4_0
out = np.zeros(total_elements, dtype=np.float32)
for i in range(num_blocks):
    start = i * GROUP_SIZE
    block = raw_bytes[start:start + GROUP_SIZE]
    scale = np.frombuffer(block[:2], dtype=np.float16)[0].astype(np.float32)
    
    for j in range(16):
        qs = block[2 + j]
        q0 = (qs & 0x0F) - 8
        q1 = (qs >> 4) - 8
        out[i * BLOCK_SIZE + j] = q0 * scale
        out[i * BLOCK_SIZE + j + 16] = q1 * scale

python_qkv = out.reshape(EXPECTED_SHAPE)
print(f"Python Q4_0 dequant shape: {python_qkv.shape}")
print(f"Python mean: {python_qkv.mean():.6f}, std: {python_qkv.std():.6f}, abs_mean: {np.abs(python_qkv).mean():.6f}")

# Load Rust dequantized
rust_qkv = np.fromfile("gguf_qkv_layer0.bin", dtype=np.float32).reshape(EXPECTED_SHAPE)
print(f"Rust Q4_0 dequant shape: {rust_qkv.shape}")
print(f"Rust mean: {rust_qkv.mean():.6f}, std: {rust_qkv.std():.6f}, abs_mean: {np.abs(rust_qkv).mean():.6f}")

# Compare
diff = python_qkv - rust_qkv
print(f"Diff abs_mean: {np.abs(diff).mean():.10f}")
print(f"Diff max: {np.abs(diff).max():.10f}")
print(f"Cosine similarity: {np.dot(python_qkv.flatten(), rust_qkv.flatten()) / (np.linalg.norm(python_qkv) * np.linalg.norm(rust_qkv)):.10f}")
