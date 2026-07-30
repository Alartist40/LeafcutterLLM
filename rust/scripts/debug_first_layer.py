"""Debug: run first linear attention layer and dump intermediate values."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import safetensors
from pathlib import Path

model_dir = Path("/home/xander/Downloads/models/ornith safetensor")
config_path = model_dir / "config.json"

# Load config
import json
with open(config_path) as f:
    config = json.load(f)
text_config = config["text_config"]

h = text_config["hidden_size"]
n_qk = text_config["linear_num_key_heads"]
n_v = text_config["linear_num_value_heads"]
d_k = text_config["linear_key_head_dim"]
d_v = text_config["linear_value_head_dim"]
conv_k = text_config["linear_conv_kernel_dim"]
qk_dim = n_qk * d_k
v_dim = n_v * d_v
conv_dim = qk_dim * 2 + v_dim

# Load all safetensors
safetensor_files = sorted(model_dir.glob("*.safetensors"))
print(f"Found {len(safetensor_files)} safetensor files")

# Read layer 0 weights
def read_tensor(name):
    for f in safetensor_files:
        with safetensors.safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    return None

# prompt tokens: "The capital of France is" -> [760, 6511, 314, 9338, 369]
tokens = [760, 6511, 314, 9338, 369]

# Get embedding
embed = read_tensor("model.language_model.embed_tokens.weight")
print(f"Embed shape: {embed.shape}")

hidden = embed[tokens[0]]  # First token: "The"
print(f"Hidden[:4] = {hidden[:4].tolist()}")
print(f"Hidden mean_abs = {hidden.abs().mean().item():.6f}")

# Layer 0 weights
norm_w = read_tensor("model.language_model.layers.0.input_layernorm.weight")
qkv_w = read_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
conv_w = read_tensor("model.language_model.layers.0.linear_attn.conv1d.weight")
a_w = read_tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight")
b_w = read_tensor("model.language_model.layers.0.linear_attn.in_proj_b.weight")
z_w = read_tensor("model.language_model.layers.0.linear_attn.in_proj_z.weight")
a_log = read_tensor("model.language_model.layers.0.linear_attn.A_log")
dt_bias = read_tensor("model.language_model.layers.0.linear_attn.dt_bias")
norm_w2 = read_tensor("model.language_model.layers.0.linear_attn.norm.weight")
out_w = read_tensor("model.language_model.layers.0.linear_attn.out_proj.weight")

# 1. Input norm
normed = hidden * norm_w / (hidden.pow(2).mean().sqrt() + 1e-6)
print(f"\nNormed[:4] = {normed[:4].tolist()}")
print(f"Normed mean_abs = {normed.abs().mean().item():.6f}")

# 2. QKV projection
qkv = normed @ qkv_w.T
print(f"\nQKV[:4] = {qkv[:4].tolist()}")
print(f"QKV mean_abs = {qkv.abs().mean().item():.6f}")

# 3. Conv1d (manual, matching PyTorch padding=same)
# conv_w shape: [conv_dim, 1, 4], tap order: [tap0(3ago), tap1(2ago), tap2(1ago), tap3(current)]
# For first token (t=0): input[t-1], input[t-2], input[t-3] are zero (padded)
# conv_out[c] = w[c,0,3]*input[t] + w[c,0,2]*0 + w[c,0,1]*0 + w[c,0,0]*0 = w[c,0,3]*input[t]
conv = conv_w[:, 0, -1] * qkv  # tap 3 (last) = newest tap = weight for current input
print(f"\nConv output[:4] = {conv[:4].tolist()}")
print(f"Conv output mean_abs = {conv.abs().mean().item():.6f}")

# 4. SiLU
conv_out = torch.nn.functional.silu(conv)
print(f"\nConv+SiLU[:4] = {conv_out[:4].tolist()}")
print(f"Conv+SiLU mean_abs = {conv_out.abs().mean().item():.6f}")

# 5. Split Q, K, V
q_data = conv_out[:qk_dim]
k_data = conv_out[qk_dim:2*qk_dim]
v_data = conv_out[2*qk_dim:]
print(f"\nQ mean_abs = {q_data.abs().mean().item():.6f}")
print(f"K mean_abs = {k_data.abs().mean().item():.6f}")
print(f"V mean_abs = {v_data.abs().mean().item():.6f}")

# 6. L2 normalize Q and K per head
q = q_data.clone()
k = k_data.clone()
for head in range(n_qk):
    base = head * d_k
    norm_q = q[base:base+d_k].norm()
    q[base:base+d_k] /= (norm_q + 1e-6)
    norm_k = k[base:base+d_k].norm()
    k[base:base+d_k] /= (norm_k + 1e-6)

# Scale Q
scale = 1.0 / (d_k ** 0.5)
q *= scale

print(f"\nQ after norm[:4] = {q[:4].tolist()}")
print(f"Q after norm mean_abs = {q.abs().mean().item():.6f}")

# 7. Decay
alpha = normed @ a_w.T
print(f"\nAlpha[:4] = {alpha[:4].tolist()}")

decays = []
for head in range(n_v):
    dt = torch.nn.functional.softplus(alpha[head] + dt_bias[head])
    a = -a_log[head].exp()
    decay = (dt * a).exp()
    decays.append(decay.item())
print(f"Decays[:8] = {decays[:8]}")

# 8. Beta
beta_logits = normed @ b_w.T
betas = torch.sigmoid(beta_logits)
print(f"Betas[:8] = {betas[:8].tolist()}")

# 9. Delta rule (first token, state=0)
# S_i = 0, v_pred = 0, S = beta * v ⊗ k, output = S @ q
v_heads_per_qk = n_v // n_qk
output = torch.zeros(v_dim)
state = {}  # We'll compute per head

for h_qk in range(n_qk):
    q_h = q[h_qk * d_k:(h_qk+1)*d_k]
    k_h = k[h_qk * d_k:(h_qk+1)*d_k]
    for v_off in range(v_heads_per_qk):
        h_v = h_qk * v_heads_per_qk + v_off
        if h_v >= n_v:
            continue
        v_h = v_data[h_v * d_v:(h_v+1)*d_v]
        decay_h = decays[h_v]
        beta_h = betas[h_v].item()
        
        # First token: state = 0, v_pred = 0
        v_pred = torch.zeros(d_v)
        # State update: S = beta * v ⊗ k
        S = beta_h * torch.outer(v_h, k_h)
        # Output: o = S @ q
        o = S @ q_h
        output[h_v * d_v:(h_v+1)*d_v] = o

print(f"\nDelta output[:8] = {output[:8].tolist()}")
print(f"Delta output mean_abs = {output.abs().mean().item():.6f}")

# 10. Per-head RMSNorm
output_rms = output.clone().view(n_v, d_v)
for h in range(n_v):
    rms = output_rms[h].pow(2).mean().sqrt()
    output_rms[h] = (output_rms[h] / (rms + 1e-6)) * norm_w2
output = output_rms.flatten()
print(f"\nAfter RMSNorm[:8] = {output[:8].tolist()}")
print(f"After RMSNorm mean_abs = {output.abs().mean().item():.6f}")

# 11. Z-gate
z = normed @ z_w.T
silu_z = z * torch.sigmoid(z)
output *= silu_z
print(f"\nZ[:8] = {z[:8].tolist()}")
print(f"Silu(z)[:8] = {silu_z[:8].tolist()}")
print(f"After z-gate[:8] = {output[:8].tolist()}")

# 12. Out proj
# out_w.dtype is BF16, need to convert
out_w_f32 = out_w.float()
result = output.to(out_w_f32.dtype) @ out_w_f32.T
print(f"\nFinal output[:8] = {result[:8].tolist()}")
print(f"Final output mean_abs = {result.abs().mean().item():.6f}")
