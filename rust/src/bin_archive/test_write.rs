fn main() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    println!("data.len()={}, bytes.len()={}", data.len(), bytes.len());
    std::fs::write("test_write.bin", bytes).expect("write");
    let meta = std::fs::metadata("test_write.bin").expect("meta");
    println!("file size={}", meta.len());
}
