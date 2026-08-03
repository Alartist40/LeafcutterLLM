use leafcutter::model::gguf::GGUFile;
use leafcutter::gguf_provider::extract_ornith_config;
fn main() {
    let gguf = GGUFile::open("/home/xander/Downloads/models/ornith-1.0-9b-Q8_0.gguf").unwrap();
    let cfg = extract_ornith_config(&gguf).unwrap();
    eprintln!("intermediate_size = {}", cfg.intermediate_size);
    eprintln!("hidden_size = {}", cfg.hidden_size);
    eprintln!("num_attention_heads = {}", cfg.num_attention_heads);
    eprintln!("num_key_value_heads = {}", cfg.num_key_value_heads);
    eprintln!("head_dim = {}", cfg.head_dim);
    eprintln!("linear_num_key_heads = {}", cfg.linear_num_key_heads);
    eprintln!("linear_num_value_heads = {}", cfg.linear_num_value_heads);
    eprintln!("linear_key_head_dim = {}", cfg.linear_key_head_dim);
    eprintln!("linear_value_head_dim = {}", cfg.linear_value_head_dim);
    eprintln!("linear_conv_kernel_dim = {}", cfg.linear_conv_kernel_dim);
}
