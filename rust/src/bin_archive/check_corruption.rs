use leafcutter::model::loader::GGUFModel;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let corruption = leafcutter::model::loader::scan_for_corruption(&model.file);
    println!("{}", corruption.print());
}
