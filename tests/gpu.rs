//! GPU correctness tests against the CPU reference. Skipped (with a message) when no CUDA device
//! is present, unless `TERNARY_REQUIRE_GPU=1` is set (scripts/bench.sh sets it), in which case a
//! missing GPU is a failure.

use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ternary_gemm::gpu::{Gpu, Launch, ROWS_PER_WARP};
use ternary_gemm::pack::PackedTernary;
use ternary_gemm::reference::{self, quantize_acts};

/// (N out, K in): BitNet-b1.58-2B (hidden 2560, ffn 6912, kv 640) and Llama-7B-like 4096/11008.
const SHAPES: &[(usize, usize)] = &[
    (2560, 2560),
    (640, 2560),
    (6912, 2560),
    (2560, 6912),
    (11008, 4096),
    (4096, 11008),
];
const BATCHES: &[usize] = &[1, 3, 16];

fn gpu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) => {
            if std::env::var("TERNARY_REQUIRE_GPU").as_deref() == Ok("1") {
                panic!("TERNARY_REQUIRE_GPU=1 but no usable GPU: {e}");
            }
            eprintln!("SKIPPED (no GPU): {e}");
            None
        }
    }
}

struct Case {
    p: PackedTernary,
    x: Vec<f16>,
    gamma: Vec<f16>,
    res: Vec<f16>,
}

fn make_case(rng: &mut StdRng, n: usize, k: usize, batch: usize) -> Case {
    let wt: Vec<i8> = (0..n * k).map(|_| rng.random_range(-1i8..=1)).collect();
    let scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.01f32..0.05)).collect();
    let p = PackedTernary::from_ternary(&wt, n, k, scales).unwrap();
    let x = (0..batch * k)
        .map(|_| f16::from_f32(rng.random_range(-2.0f32..2.0)))
        .collect();
    let gamma = (0..k)
        .map(|_| f16::from_f32(rng.random_range(0.5f32..1.5)))
        .collect();
    let res = (0..batch * n)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
        .collect();
    Case { p, x, gamma, res }
}

fn count_bit_mismatch(a: &[f16], b: &[f16]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn int8_path_is_bit_exact() {
    let Some(g) = gpu() else { return };
    let mut rng = StdRng::seed_from_u64(42);
    for &(n, k) in SHAPES {
        for &batch in BATCHES {
            let c = make_case(&mut rng, n, k, batch);
            let w = g.upload_weights(&c.p).unwrap();
            let x = g.upload_f16(&c.x).unwrap();
            let res = g.upload_f16(&c.res).unwrap();
            let mut acts = g.alloc_acts(batch, k).unwrap();
            g.quantize(&x, None, 1e-5, &mut acts).unwrap();

            // 1. GPU quantization == CPU quantization (bitwise)
            let cpu_a = quantize_acts(&c.x, batch, k, None, 1e-5);
            assert_eq!(
                g.download(&acts.q).unwrap(),
                cpu_a.q,
                "quantize q {n}x{k} b{batch}"
            );
            assert_eq!(g.download(&acts.s).unwrap(), cpu_a.s, "quantize s");
            assert_eq!(g.download(&acts.sum).unwrap(), cpu_a.sum, "quantize sum");

            for with_res in [false, true] {
                let r = with_res.then_some(&res);
                let expect = reference::gemm_i8(&c.p, &cpu_a, with_res.then_some(&c.res[..]));
                // 2. optimized kernel, every rows-per-warp variant
                for rpw in ROWS_PER_WARP {
                    let mut out = g.alloc_f16(batch * n).unwrap();
                    g.gemm_i8(
                        &w,
                        &acts,
                        r,
                        &mut out,
                        Launch {
                            rows_per_warp: Some(rpw),
                            warps: 4,
                        },
                    )
                    .unwrap();
                    let got = g.download(&out).unwrap();
                    assert_eq!(
                        count_bit_mismatch(&got, &expect),
                        0,
                        "ternary_i8 {n}x{k} b{batch} r{rpw} res={with_res}"
                    );
                }
                // 3. fused single-launch kernel
                if batch * k <= ternary_gemm::gpu::FUSED_SMEM_LIMIT {
                    let mut out = g.alloc_f16(batch * n).unwrap();
                    g.gemm_i8_fused(&w, &x, batch, None, 1e-5, r, &mut out, Launch::default())
                        .unwrap();
                    let got = g.download(&out).unwrap();
                    assert_eq!(
                        count_bit_mismatch(&got, &expect),
                        0,
                        "fused {n}x{k} b{batch} res={with_res}"
                    );
                }
                // 4. naive kernel
                let mut out = g.alloc_f16(batch * n).unwrap();
                g.naive_i8(&w, &acts, r, &mut out).unwrap();
                let got = g.download(&out).unwrap();
                assert_eq!(
                    count_bit_mismatch(&got, &expect),
                    0,
                    "naive {n}x{k} b{batch} res={with_res}"
                );
            }
        }
    }
}

#[test]
fn fused_rmsnorm_matches_within_tolerance() {
    let Some(g) = gpu() else { return };
    let mut rng = StdRng::seed_from_u64(43);
    for &(n, k) in SHAPES {
        for &batch in &[1usize, 4] {
            if batch * k > ternary_gemm::gpu::FUSED_SMEM_LIMIT {
                continue;
            }
            let c = make_case(&mut rng, n, k, batch);
            let w = g.upload_weights(&c.p).unwrap();
            let x = g.upload_f16(&c.x).unwrap();
            let gam = g.upload_f16(&c.gamma).unwrap();
            let mut out = g.alloc_f16(batch * n).unwrap();
            g.gemm_i8_fused(
                &w,
                &x,
                batch,
                Some(&gam),
                1e-5,
                None,
                &mut out,
                Launch::default(),
            )
            .unwrap();
            let got = g.download(&out).unwrap();
            // GPU rsqrtf + different summation order can flip a few int8 roundings by 1 LSB.
            let cpu_a = quantize_acts(&c.x, batch, k, Some(&c.gamma), 1e-5);
            let expect = reference::gemm_i8(&c.p, &cpu_a, None);
            let max_ref = expect.iter().map(|v| v.to_f32().abs()).fold(0f32, f32::max);
            let max_err = got
                .iter()
                .zip(&expect)
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0f32, f32::max);
            assert!(
                max_err <= 1e-2 * max_ref + 1e-3,
                "rmsnorm fused {n}x{k} b{batch}: err {max_err} vs max {max_ref}"
            );
            // and the separate quantize kernel with gamma gives the same int8 as the fused one would
            let mut acts = g.alloc_acts(batch, k).unwrap();
            g.quantize(&x, Some(&gam), 1e-5, &mut acts).unwrap();
            let q = g.download(&acts.q).unwrap();
            let off = q
                .iter()
                .zip(&cpu_a.q)
                .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1)
                .count();
            assert_eq!(off, 0, "rmsnorm quantize differs by >1 LSB");
            let mut out2 = g.alloc_f16(batch * n).unwrap();
            g.gemm_i8(&w, &acts, None, &mut out2, Launch::default())
                .unwrap();
            // block sizes differ (so the sum-of-squares order differs): compare with the same tolerance
            let got2 = g.download(&out2).unwrap();
            let max_err2 = got2
                .iter()
                .zip(&expect)
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0f32, f32::max);
            assert!(
                max_err2 <= 1e-2 * max_ref + 1e-3,
                "rmsnorm quantize+gemm {n}x{k} b{batch}: err {max_err2}"
            );
        }
    }
}

#[test]
fn f16_path_matches_within_tolerance() {
    let Some(g) = gpu() else { return };
    let mut rng = StdRng::seed_from_u64(44);
    for &(n, k) in SHAPES {
        for &batch in BATCHES {
            let c = make_case(&mut rng, n, k, batch);
            let w = g.upload_weights(&c.p).unwrap();
            let x = g.upload_f16(&c.x).unwrap();
            let res = g.upload_f16(&c.res).unwrap();
            let expect = reference::gemm_f16(&c.p, &c.x, batch, Some(&c.res));
            let max_ref = expect.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let check = |got: Vec<f16>, what: &str| {
                let max_err = got
                    .iter()
                    .zip(&expect)
                    .map(|(a, b)| (a.to_f32() - b).abs())
                    .fold(0f32, f32::max);
                // fp16 output rounding (2^-11 relative) dominates; fp32 accumulation error is far smaller.
                assert!(
                    max_err <= 2e-3 * max_ref + 1e-3,
                    "{what} {n}x{k} b{batch}: err {max_err} vs max {max_ref}"
                );
            };
            for rpw in ROWS_PER_WARP {
                let mut out = g.alloc_f16(batch * n).unwrap();
                g.gemm_f16(
                    &w,
                    &x,
                    batch,
                    Some(&res),
                    &mut out,
                    Launch {
                        rows_per_warp: Some(rpw),
                        warps: 4,
                    },
                )
                .unwrap();
                check(g.download(&out).unwrap(), &format!("ternary_f16 r{rpw}"));
            }
            let mut out = g.alloc_f16(batch * n).unwrap();
            g.naive_f16(&w, &x, batch, Some(&res), &mut out).unwrap();
            check(g.download(&out).unwrap(), "naive_f16");
        }
    }
}

#[test]
fn cublas_baseline_is_sane() {
    let Some(g) = gpu() else { return };
    let cb = ternary_gemm::gpu::Cublas::new(&g).unwrap();
    let mut rng = StdRng::seed_from_u64(45);
    let (n, k, batch) = (640, 2560, 3);
    let c = make_case(&mut rng, n, k, batch);
    // dense fp16 copy of the ternary weights (times row scale)
    let wd: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(c.p.get(i / k, i % k) as f32 * c.p.row_scale[i / k]))
        .collect();
    let w = g.upload_f16(&wd).unwrap();
    let x = g.upload_f16(&c.x).unwrap();
    let mut out = g.alloc_f16(batch * n).unwrap();
    cb.gemm(&w, &x, &mut out, n, k, batch).unwrap();
    let got = g.download(&out).unwrap();
    let expect = reference::gemm_f16(&c.p, &c.x, batch, None);
    let max_ref = expect.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let max_err = got
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a.to_f32() - b).abs())
        .fold(0f32, f32::max);
    assert!(max_err <= 1e-2 * max_ref + 1e-3, "cublas err {max_err}");
}

#[test]
fn rejects_bad_shapes() {
    let Some(g) = gpu() else { return };
    let w = g.random_weights(128, 256, 1).unwrap();
    let acts = g.alloc_acts(17, 256).unwrap();
    let mut out = g.alloc_f16(17 * 128).unwrap();
    assert!(g
        .gemm_i8(&w, &acts, None, &mut out, Launch::default())
        .is_err()); // batch > 16
    let acts = g.alloc_acts(2, 256).unwrap();
    let mut small = g.alloc_f16(10).unwrap();
    assert!(g
        .gemm_i8(&w, &acts, None, &mut small, Launch::default())
        .is_err()); // out too small
}
