//! Test: load Ornith config from HuggingFace config.json.
use leafcutter::ornith_config::OrnithConfig;

fn main() {
    let path = "/home/xander/Downloads/models/ornith safetensor/config.json";
    println!("Loading Ornith config from {path}");
    let cfg = OrnithConfig::load(path).unwrap();

    println!("\nOrnith-1.0-9B Configuration:");
    println!("  hidden_size:          {}", cfg.hidden_size);
    println!("  num_hidden_layers:    {}", cfg.num_hidden_layers);
    println!("  num_attention_heads:  {}", cfg.num_attention_heads);
    println!("  num_key_value_heads:  {}", cfg.num_key_value_heads);
    println!("  head_dim:             {}", cfg.head_dim);
    println!("  intermediate_size:    {}", cfg.intermediate_size);
    println!("  vocab_size:           {}", cfg.vocab_size);
    println!("  rms_norm_eps:         {}", cfg.rms_norm_eps);
    println!("  rope_theta:           {}", cfg.rope_theta);
    println!("  linear_num_key_heads: {}", cfg.linear_num_key_heads);
    println!("  linear_num_val_heads: {}", cfg.linear_num_value_heads);
    println!("  linear_key_head_dim:  {}", cfg.linear_key_head_dim);
    println!("  linear_val_head_dim:  {}", cfg.linear_value_head_dim);
    println!("  linear_conv_kernel:   {}", cfg.linear_conv_kernel_dim);
    println!("  max_position_embeds:  {}", cfg.max_position_embeddings);

    println!("\nLayer types (32 layers):");
    let counts: std::collections::HashMap<String, usize> = cfg
        .layer_types
        .iter()
        .fold(std::collections::HashMap::new(), |mut acc, t| {
            *acc.entry(t.clone()).or_insert(0) += 1;
            acc
        });
    for (t, n) in counts {
        println!("  {t}: {n}");
    }
}
