use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let layer = std::env::args().nth(2).unwrap().parse::<usize>().unwrap();
    let model = leafcutter::model::loader::GGUFModel::load(&path).unwrap();
    let weights = model.load_layer(layer).unwrap();
    let mut names: Vec<_> = weights.keys().collect();
    names.sort();
    for name in names {
        println!("{}", name);
    }
}
