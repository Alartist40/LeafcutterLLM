fn main() {
    let m = 16usize;
    let n = 16usize;
    let k = 16usize;
    let mut r = vec![0.0f32; m * n];
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.0,
            r.as_mut_ptr(), n as isize, 1,
        );
    }
    println!("sgemm: max={}", r.iter().fold(0.0f32, |a, &b| a.max(b)));
}
