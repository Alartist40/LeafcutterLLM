import sys
sys.path.insert(0, '/home/xander/Documents/portfolio/LeafcutterLLM/venv/lib/python3.14/site-packages')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/Qwen3.5-0.8B-HF"
tokens = [17, 10, 17, 28]

print("Loading HF model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([tokens])
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

logits = outputs.logits[0, -1, :].numpy()
top_idx = int(np.argmax(logits))
print(f"HF top token: {top_idx} (logit={logits[top_idx]:.2f})")

# Save per-layer hidden states
for i, hidden in enumerate(outputs.hidden_states):
    h = hidden[0, -1, :].numpy().astype(np.float32)
    h.tofile(f"hf_layer_{i:02d}.bin")
    
print(f"Saved {len(outputs.hidden_states)} layer hidden states")

# Also print token names
for tid in [top_idx, 12, 19, 644, 6205]:
    print(f"Token {tid} = '{tokenizer.decode([tid])}'")
