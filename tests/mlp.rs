//! Signal-model (small ternary MLP) correctness on the CPU: the SIMD engine and a lane-level
//! transliteration of the `ternary_mlp` CUDA kernel must match the scalar reference bit for bit.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ternary_gemm::mlp::{quantize_into, Act, CpuEngine, DenseF32Mlp, MlpLayer, TernaryMlp};

const MODELS: &[&[usize]] = &[
    &[16, 16],
    &[64, 256, 256, 1],
    &[48, 96, 8],
    &[256, 1024, 1024, 16],
    &[1024, 1024, 1024, 1024, 4],
];
const ACTS: &[Act] = &[Act::Relu, Act::Relu2, Act::Identity];

fn inputs(rng: &mut StdRng, d: usize) -> Vec<Vec<f32>> {
    vec![
        (0..d).map(|_| rng.random_range(-3.0f32..3.0)).collect(),
        (0..d).map(|_| rng.random_range(-1e-3f32..1e-3)).collect(),
        (0..d)
            .map(|i| if i % 7 == 0 { 1e4 } else { -0.5 })
            .collect(),
        vec![0.0; d],
    ]
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Rust transliteration of `mlp_forward` in kernels/ternary.cu: k-major words, packed int8
/// activation words, dp4a over code bytes, `- sum_q` offset, `_rn` epilogue.
fn emulate_kernel(m: &TernaryMlp, x: &[f32]) -> Vec<f32> {
    fn dp4a(a: u32, b: u32, c: i32) -> i32 {
        (0..4).fold(c, |acc, i| {
            acc + ((a >> (8 * i)) as u8 as i8 as i32) * ((b >> (8 * i)) as u8 as i8 as i32)
        })
    }
    let mut cur = x.to_vec();
    for l in &m.layers {
        let words = l.kmajor_words();
        let mut q = vec![0i8; l.k];
        let (s, sum_q) = quantize_into(&cur, &mut q);
        let qw: Vec<u32> = q
            .chunks(4)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |a, (j, &v)| a | ((v as u8 as u32) << (8 * j)))
            })
            .collect();
        let mut out = vec![0f32; l.n];
        for r in 0..l.n {
            let mut acc = 0i32;
            for w in 0..l.k / 16 {
                let word = words[w * l.n + r];
                for j in 0..4 {
                    acc = dp4a((word >> (2 * j)) & 0x0303_0303, qw[4 * w + j], acc);
                }
            }
            let v = ((acc - sum_q) as f32 / s) * l.scale[r] + l.bias[r];
            out[r] = match l.act {
                Act::Identity => v,
                Act::Relu => {
                    if v > 0.0 {
                        v
                    } else {
                        0.0
                    }
                }
                Act::Relu2 => {
                    let t = if v > 0.0 { v } else { 0.0 };
                    t * t
                }
            };
        }
        cur = out;
    }
    cur
}

#[test]
fn cpu_engine_and_kernel_emulation_are_bit_exact() {
    let mut rng = StdRng::seed_from_u64(7);
    for (mi, dims) in MODELS.iter().enumerate() {
        for &a in ACTS {
            let m = TernaryMlp::random(dims, a, 100 + mi as u64).unwrap();
            let mut eng = CpuEngine::new(&m).unwrap();
            let mut bytes = CpuEngine::new(&m).unwrap().without_packed();
            let mut out = vec![0f32; m.out_dim()];
            for x in inputs(&mut rng, m.in_dim()) {
                let r = m.forward_reference(&x);
                eng.forward_into(&x, &mut out);
                assert_eq!(bits(&r), bits(&out), "CpuEngine {dims:?} {a:?}");
                bytes.forward_into(&x, &mut out);
                assert_eq!(bits(&r), bits(&out), "CpuEngine byte codes {dims:?} {a:?}");
                assert_eq!(
                    bits(&r),
                    bits(&emulate_kernel(&m, &x)),
                    "kernel emulation {dims:?} {a:?}"
                );
            }
        }
    }
}

#[test]
fn dense_f32_baseline_is_close() {
    // The dense baseline skips int8 activation quantization, so it only matches approximately.
    let m = TernaryMlp::random(&[64, 256, 256, 4], Act::Relu, 3).unwrap();
    let mut d = DenseF32Mlp::new(&m);
    let mut rng = StdRng::seed_from_u64(9);
    let x: Vec<f32> = (0..64).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let r = m.forward_reference(&x);
    let mut o = vec![0f32; 4];
    d.forward_into(&x, &mut o);
    let scale = r.iter().fold(1e-3f32, |a, v| a.max(v.abs()));
    for (a, b) in r.iter().zip(&o) {
        assert!((a - b).abs() <= 0.1 * scale, "{a} vs {b}");
    }
}

#[test]
fn validation_rejects_bad_models() {
    let bad_k = TernaryMlp {
        layers: vec![MlpLayer {
            n: 4,
            k: 10,
            w: vec![0; 40],
            scale: vec![1.0; 4],
            bias: vec![0.0; 4],
            act: Act::Relu,
        }],
    };
    assert!(bad_k.validate().is_err());
    let mut chain = TernaryMlp::random(&[16, 32, 4], Act::Relu, 1).unwrap();
    chain.layers[1].k = 48;
    assert!(chain.validate().is_err());
    assert!(TernaryMlp::random(&[16, 8192], Act::Relu, 1).is_err());
    let mut nt = TernaryMlp::random(&[16, 16], Act::Relu, 1).unwrap();
    nt.layers[0].w[0] = 2;
    assert!(nt.validate().is_err());
}

#[test]
fn engine_reports_simd_path() {
    let m = TernaryMlp::random(&[64, 64], Act::Relu, 1).unwrap();
    let e = CpuEngine::new(&m).unwrap();
    let s = e.simd();
    assert!(
        [
            "portable",
            "avx2",
            "avx512-vnni",
            "avx512-vnni-vbmi packed2",
            "avx2 packed2"
        ]
        .contains(&s),
        "{s}"
    );
    if cfg!(not(any(feature = "avx2", feature = "avx512"))) {
        assert_eq!(s, "portable");
    }
    eprintln!("CpuEngine SIMD path on this machine: {s}");
}

#[test]
fn simd_quantize_matches_scalar_oracle() {
    use ternary_gemm::mlp::quantize_scalar;
    let mut rng = StdRng::seed_from_u64(11);
    for k in [1usize, 7, 8, 31, 32, 33, 64, 100, 255, 256, 1024, 4096] {
        for x in inputs(&mut rng, k) {
            let (mut a, mut b) = (vec![0i8; k], vec![0i8; k]);
            let ra = quantize_into(&x, &mut a);
            let rb = quantize_scalar(&x, &mut b);
            assert_eq!((ra.0.to_bits(), ra.1), (rb.0.to_bits(), rb.1), "k={k}");
            assert_eq!(a, b, "k={k}");
        }
        // exact .5 ties must round to even in every path
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - k as f32 / 2.0) * 0.5).collect();
        let (mut a, mut b) = (vec![0i8; k], vec![0i8; k]);
        assert_eq!(quantize_into(&x, &mut a).1, quantize_scalar(&x, &mut b).1);
        assert_eq!(a, b);
    }
}
