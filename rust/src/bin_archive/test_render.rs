use leafcutter::model::gguf::GGUFile;
use leafcutter::profiles::{render_prompt, resolve_profile};
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let profile = resolve_profile(&file.metadata, None);
    println!("[profile] {} desc={}", profile.name, profile.description);
    let rendered = render_prompt(&profile, "", "hi");
    println!("---RENDERED PROMPT---");
    println!("{}", rendered);
    println!("---END---");
}
