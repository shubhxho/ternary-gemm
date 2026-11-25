//! Property tests (proptest): packing, quantization and GEMM against independent dense oracles.

use half::f16;
use proptest::prelude::*;
use ternary_gemm::cpu::CpuTernary;
use ternary_gemm::mlp::{quantize_into, quantize_scalar, Act, CpuEngine, TernaryMlp};
use ternary_gemm::pack::{PackError, PackedTernary, K_ALIGN};
use ternary_gemm::reference::{gemm_i8, quantize_acts};

fn ternary_matrix() -> impl Strategy<Value = (usize, usize, Vec<i8>)> {
    (1usize..6, 1usize..40).prop_flat_map(|(n, kb)| {
        let k = kb * K_ALIGN;
        (Just(n), Just(k), proptest::collection::vec(-1i8..=1, n * k))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// unpack(pack(w)) == w, and both layouts decode to the same weights.
    #[test]
    fn pack_roundtrip((n, k, w) in ternary_matrix()) {
        let p = PackedTernary::from_ternary(&w, n, k, vec![1.0; n]).unwrap();
        prop_assert_eq!(p.unpack(), w.clone());
        for r in 0..n {
            for c in 0..k {
                prop_assert_eq!(p.get(r, c), p.get_natural(r, c));
            }
        }
        let zeros = w.iter().filter(|&&v| v == 0).count() as f64 / (n * k) as f64;
        prop_assert!((p.sparsity() - zeros).abs() < 1e-12);
    }

    /// Any K that is not a positive multiple of 64 is rejected with KNotAligned.
    #[test]
    fn pack_rejects_misaligned_k(k in 1usize..512) {
        prop_assume!(k % K_ALIGN != 0);
        let r = PackedTernary::from_ternary(&vec![0; k], 1, k, vec![1.0]);
        prop_assert_eq!(r.unwrap_err(), PackError::KNotAligned(k));
    }

    /// Quantized activations stay in range, sum is consistent, and dequantization error is
    /// bounded by half a quantization step.
    #[test]
    fn quantize_properties(x in proptest::collection::vec(-1e3f32..1e3, 1..300)) {
        let mut q = vec![0i8; x.len()];
        let (s, sum) = quantize_scalar(&x, &mut q);
        prop_assert_eq!(sum, q.iter().map(|&v| v as i32).sum::<i32>());
        prop_assert!(q.iter().all(|&v| v >= -127));
        for (&xi, &qi) in x.iter().zip(&q) {
            prop_assert!((xi * s - qi as f32).abs() <= 0.5 + 1e-3);
        }
        let mut q2 = vec![0i8; x.len()];
        prop_assert_eq!(quantize_into(&x, &mut q2), (s, sum));
        prop_assert_eq!(q2, q);
    }

    /// The fast CPU activation quantizer is bit-identical to the reference, with and without RMSNorm.
    #[test]
    fn fast_quantize_acts_bit_exact(
        x in proptest::collection::vec(-6e4f32..6e4, 1..400),
        batch in 1usize..4,
        g in 0.01f32..4.0,
    ) {
        let k = x.len() / batch;
        prop_assume!(k > 0);
        let xs: Vec<f16> = x[..batch * k].iter().map(|&v| f16::from_f32(v)).collect();
        let gamma: Vec<f16> = (0..k).map(|i| f16::from_f32(g + i as f32 * 1e-3)).collect();
        for gm in [None, Some(&gamma[..])] {
            prop_assert_eq!(
                ternary_gemm::cpu::quantize_acts(&xs, batch, k, gm, 1e-5),
                quantize_acts(&xs, batch, k, gm, 1e-5)
            );
        }
    }

    /// Reference GEMM equals an independent dense i32 GEMM with the same epilogue, and the CPU
    /// backend equals the reference.
    #[test]
    fn gemm_matches_dense_oracle(
        (n, k, w) in ternary_matrix(),
        batch in 1usize..20,
        seed in any::<u64>(),
    ) {
        let mut st = seed | 1;
        let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0 };
        let x: Vec<f16> = (0..batch * k).map(|_| f16::from_f32(rnd())).collect();
        let scales: Vec<f32> = (0..n).map(|i| 0.01 + i as f32 * 0.003).collect();
        let p = PackedTernary::from_ternary(&w, n, k, scales.clone()).unwrap();
        let a = quantize_acts(&x, batch, k, None, 1e-5);
        let y = gemm_i8(&p, &a, None);
        for b in 0..batch {
            for r in 0..n {
                let acc: i32 = (0..k).map(|c| w[r * k + c] as i32 * a.q[b * k + c] as i32).sum();
                let want = f16::from_f32(acc as f32 / a.s[b] * scales[r]);
                prop_assert_eq!(y[b * n + r].to_bits(), want.to_bits());
            }
        }
        let cpu = CpuTernary::new(&p).with_threads(3).gemm_i8(&a, None);
        prop_assert_eq!(cpu, y);
    }

    /// The SIMD signal-model engine is bit-exact with the reference for random widths.
    #[test]
    fn mlp_engine_bit_exact(
        widths in proptest::collection::vec(1usize..20, 2..5),
        seed in any::<u64>(),
        xs in proptest::collection::vec(-5f32..5.0, 320),
    ) {
        let dims: Vec<usize> = widths.iter().enumerate()
            .map(|(i, w)| if i + 1 == widths.len() { *w } else { w * 16 })
            .collect();
        let m = TernaryMlp::random(&dims, Act::Relu2, seed).unwrap();
        let x = &xs[..m.in_dim()];
        let mut e = CpuEngine::new(&m).unwrap();
        let mut out = vec![0f32; m.out_dim()];
        e.forward_into(x, &mut out);
        let want = m.forward_reference(x);
        prop_assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut b = CpuEngine::new(&m).unwrap().without_packed();
        b.forward_into(x, &mut out);
        prop_assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
