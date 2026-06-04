#!/usr/bin/env python3
import sys
try:
    from gguf import GGUFReader
except ImportError:
    print("pip install gguf")
    sys.exit(1)

path = sys.argv[1]
reader = GGUFReader(path)

print("=== General Metadata ===")
for key in ['general.architecture', 'general.name', 'general.quantization_version',
            'general.file_type', 'general.source', 'general.converter',
            'general.unsloth_version', 'general.unsloth_model_name']:
    try:
        val = reader.get_field(key)
        print(f"{key}: {val.parts if hasattr(val, 'parts') else val}")
    except:
        print(f"{key}: NOT FOUND")

print("\n=== Architecture-Specific ===")
arch = None
try:
    arch = reader.get_field('general.architecture').parts[0].decode()
    print(f"Architecture: {arch}")
except:
    print("Architecture: NOT FOUND")

if arch and 'qwen' in arch.lower():
    for key in reader.fields.keys():
        if 'qwen' in key.lower() or 'rope' in key.lower() or 'head' in key.lower():
            try:
                val = reader.get_field(key)
                print(f"{key}: {val.parts if hasattr(val, 'parts') else val}")
            except:
                pass

print("\n=== Tensor Inventory (first 30) ===")
for i, tensor in enumerate(reader.tensors[:30]):
    print(f"{i}: {tensor.name} | shape={tensor.shape} | dtype={tensor.data.dtype if hasattr(tensor.data, 'dtype') else 'N/A'} | n_elements={tensor.data.nbytes if hasattr(tensor.data, 'nbytes') else 'N/A'}")

print("\n=== Attention/SSM Tensors ===")
for tensor in reader.tensors:
    if any(x in tensor.name for x in ['attn', 'ssm', 'qkv', 'gate', 'norm']):
        print(f"{tensor.name} | shape={tensor.shape}")
