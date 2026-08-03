use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_all_layers <gguf>");
    let model = GGUFModel::load(&path).expect("load");
    for i in 0..model.config.num_hidden_layers {
        match model.load_layer(i) {
            Ok(layer) => {
                let has_pre = layer.contains_key("input_layernorm.weight") || layer.contains_key("attn_norm.weight");
                let has_post = layer.contains_key("post_attention_layernorm.weight") || layer.contains_key("ffn_norm.weight");
                if !has_pre || !has_post {
                    println!("Layer {} MISSING: pre={} post={}", i, has_pre, has_post);
                }
            }
            Err(e) => println!("Layer {} FAILED: {:?}", i, e),
        }
    }
    println!("Done checking {} layers", model.config.num_hidden_layers);
}
