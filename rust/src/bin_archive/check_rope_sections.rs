use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    
    if let Some(v) = file.metadata.get("qwen35.rope.dimension_sections") {
        println!("qwen35.rope.dimension_sections: {:?}", v);
    } else {
        println!("qwen35.rope.dimension_sections: NOT FOUND");
    }
    
    if let Some(v) = file.metadata.get("llama.rope.dimension_sections") {
        println!("llama.rope.dimension_sections: {:?}", v);
    } else {
        println!("llama.rope.dimension_sections: NOT FOUND");
    }
    
    if let Some(v) = file.metadata.get("general.architecture") {
        println!("general.architecture: {:?}", v);
    }
}
