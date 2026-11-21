//! Dev tool: read a ternary matrix (one int per line, rows*cols values,
//! then "rows cols" on the last line) from stdin and print the tile
//! bytes as hex. Used by tools/test_ctxb_packing.py to cross-check the
//! Python packer against the Rust packer.
use cortex_kernels::BitLinearWeights;
use std::io::Read;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut nums: Vec<i64> = text
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let cols = nums.pop().unwrap() as usize;
    let rows = nums.pop().unwrap() as usize;
    let w: Vec<i8> = nums.into_iter().map(|v| v as i8).collect();
    assert_eq!(w.len(), rows * cols);
    let scales = vec![0.25f32; rows];
    let m = BitLinearWeights::from_ternary_matrix(&w, rows, cols, &scales);
    let tiles = m.tile_bytes();
    for b in tiles {
        print!("{b:02x}");
    }
    println!();
}
