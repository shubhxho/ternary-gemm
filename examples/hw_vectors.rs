//! Generate test vectors for the FPGA ternary MAC (`hw/`): weights packed with the crate's own
//! natural layout (`PackedTernary`), random int8 activations, and expected row sums computed from
//! the *unpacked* weights, so the RTL testbench checks both the arithmetic and the bit layout.
//!
//! Usage: `cargo run --release --example hw_vectors -- [out_dir] [rows] [k]` (defaults: hw/vectors 256 128)

use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::fmt::Write as _;
use ternary_gemm::PackedTernary;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().map_or("hw/vectors", |s| s.as_str());
    let rows: usize = args.get(1).map_or(256, |s| s.parse().unwrap());
    let k: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let kw = k / 16;
    let mut rng = StdRng::seed_from_u64(2026);
    // Edge rows first: all -1 x -128 (most negative), all +1 x -128, all 0, alternating; then random.
    let mut w: Vec<i8> = Vec::with_capacity(rows * k);
    let mut acts: Vec<Vec<i8>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let (wr, ar): (Vec<i8>, Vec<i8>) = match r {
            0 => (vec![-1; k], vec![-128; k]),
            1 => (vec![1; k], vec![-128; k]),
            2 => (vec![1; k], vec![127; k]),
            3 => (vec![0; k], (0..k).map(|_| rng.random()).collect()),
            4 => (
                (0..k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(),
                vec![-128; k],
            ),
            _ => (
                (0..k).map(|_| rng.random_range(-1i8..=1)).collect(),
                (0..k).map(|_| rng.random()).collect(),
            ),
        };
        w.extend(wr);
        acts.push(ar);
    }
    let p = PackedTernary::from_ternary(&w, rows, k, vec![1.0; rows]).expect("pack");
    let (mut wh, mut ah, mut eh) = (String::new(), String::new(), String::new());
    for r in 0..rows {
        let mut acc = 0i32;
        for c in 0..k {
            acc += w[r * k + c] as i32 * acts[r][c] as i32;
        }
        writeln!(eh, "{:08x}", acc as u32).unwrap();
        for j in 0..kw {
            writeln!(wh, "{:08x}", p.natural[r * kw + j]).unwrap();
            // 16 bytes, activation t in bits [8t+7:8t] -> most significant byte printed first.
            let mut line = String::new();
            for t in (0..16).rev() {
                write!(line, "{:02x}", acts[r][16 * j + t] as u8).unwrap();
            }
            writeln!(ah, "{line}").unwrap();
        }
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(format!("{dir}/w.hex"), wh).unwrap();
    std::fs::write(format!("{dir}/a.hex"), ah).unwrap();
    std::fs::write(format!("{dir}/exp.hex"), eh).unwrap();
    std::fs::write(
        format!("{dir}/params.vh"),
        format!("localparam ROWS = {rows};\nlocalparam WORDS = {kw};\n"),
    )
    .unwrap();
    println!("wrote {rows} rows x {k} (K) to {dir}/{{w,a,exp}}.hex + params.vh");
}
