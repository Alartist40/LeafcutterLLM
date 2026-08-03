fn main() {
    let path = std::env::args().nth(1).unwrap();
    let model = leafcutter::model::loader::GGUFModel::load(&path).unwrap();
    println!("norm_eps = {:?}", model.config.norm_eps);
    println!("hidden_size = {}", model.config.hidden_size);
    println!("num_hidden_layers = {}", model.config.num_hidden_layers);
    println!("intermediate_size = {}", model.config.intermediate_size);
    println!("num_attention_heads = {}", model.config.num_attention_heads);
    println!("num_key_value_heads = {}", model.config.num_key_value_heads);
}
