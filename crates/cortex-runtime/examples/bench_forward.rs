use cortex_runtime::{CortexConfig, CortexModel};
use std::time::Instant;
fn main() {
    let mut m = CortexModel::new_toy(CortexConfig::micro(), 42);
    m.hippo_learning = false;
    let t0 = Instant::now();
    let n = 200;
    for i in 0..n {
        let _ = m.forward_token((65 + (i % 26)) as u32, false);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{n} forward_tokens in {ms:.0} ms ({:.2} ms/token)",
        ms / n as f64
    );
}
