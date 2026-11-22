//! Test vectors for the non-Rust implementations under `langs/` (Zig, Triton, Mojo, Julia,
//! Futhark, Halide): each case is a ternary weight matrix in the crate's natural 2-bit layout,
//! one int8-quantized activation vector, and the outputs `reference::gemm_i8` produces.
//!
//! Usage: `cargo run --release --example lang_vectors -- [out_dir]` (default `langs/vectors`).
//!
//! File format (`case<i>.txt`, whitespace separated, one field per line):
//! ```text
//! n k
//! s_bits sum                 # f32 activation scale as hex bits, i32 sum(q)
//! row_scale_bits * n         # f32 hex bits
//! natural_words * (n*k/16)   # u32 hex; weight c of a row at word c/16, bits 2*(c%16), code w+1
//! q * k                      # int8 activations
//! acc * n                    # expected sum_k w[r][k] * q[k]
//! out_bits * n               # expected fp16 bits of acc / s * row_scale (round to nearest even)
//! ```

use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fmt::Write as _;
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::{gemm_i8, quantize_acts};

/// (n, k) per case: tiny edge shapes, a K that is not a multiple of 256, and one big case.
pub const CASES: &[(usize, usize)] = &[(1, 64), (7, 128), (64, 256), (33, 1088), (256, 4096)];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "langs/vectors".into());
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = StdRng::seed_from_u64(1729);
    for (i, &(n, k)) in CASES.iter().enumerate() {
        let w: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
        let sc: Vec<f32> = (0..n).map(|_| rng.random_range(0.005f32..0.2)).collect();
        let p = PackedTernary::from_ternary(&w, n, k, sc).unwrap();
        let x: Vec<f16> = (0..k)
            .map(|_| f16::from_f32(rng.random_range(-3.0..3.0)))
            .collect();
        let a = quantize_acts(&x, 1, k, None, 1e-5);
        let y = gemm_i8(&p, &a, None);
        let acc: Vec<i32> = (0..n)
            .map(|r| (0..k).map(|c| w[r * k + c] as i32 * a.q[c] as i32).sum())
            .collect();
        let mut s = String::new();
        let hex = |v: &mut String, it: &mut dyn Iterator<Item = u32>| {
            let parts: Vec<String> = it.map(|b| format!("{b:08x}")).collect();
            writeln!(v, "{}", parts.join(" ")).unwrap();
        };
        writeln!(s, "{n} {k}").unwrap();
        writeln!(s, "{:08x} {}", a.s[0].to_bits(), a.sum[0]).unwrap();
        hex(&mut s, &mut p.row_scale.iter().map(|v| v.to_bits()));
        hex(&mut s, &mut p.natural.iter().copied());
        let qs: Vec<String> = a.q.iter().map(|v| v.to_string()).collect();
        writeln!(s, "{}", qs.join(" ")).unwrap();
        let accs: Vec<String> = acc.iter().map(|v| v.to_string()).collect();
        writeln!(s, "{}", accs.join(" ")).unwrap();
        let ob: Vec<String> = y.iter().map(|v| format!("{:04x}", v.to_bits())).collect();
        writeln!(s, "{}", ob.join(" ")).unwrap();
        let path = format!("{dir}/case{i}.txt");
        std::fs::write(&path, s).unwrap();
        println!("{path}: n={n} k={k}");
    }
}
