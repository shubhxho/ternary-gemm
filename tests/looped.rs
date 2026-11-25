//! Looped transformer: shapes, determinism, loop=1 parity with a single block, unrolled
//! equivalence, halting, injection modes and CPU-backend parity with the reference GEMM.

use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use ternary_gemm::looped::*;
use ternary_gemm::reference::{self, QuantizedActs};

const CFG: BlockConfig = BlockConfig {
    d_model: 128,
    n_heads: 4,
    d_ff: 256,
};

fn input(tokens: usize, d: usize, seed: u64) -> Vec<f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..tokens * d)
        .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
        .collect()
}

fn lc(loops: usize, inject: Inject) -> LoopConfig {
    LoopConfig {
        loops,
        inject,
        halting: None,
    }
}

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn output_shape_and_finiteness() {
    for inject in [Inject::None, Inject::Add, Inject::Adapter] {
        let m = LoopedTransformer::random(CFG, lc(4, inject), 1).unwrap();
        for tokens in [1, 5, 17] {
            let out = m.forward_cpu(&input(tokens, CFG.d_model, 2), tokens);
            assert_eq!(out.hidden.len(), tokens * CFG.d_model);
            assert_eq!(out.loops_run, 4);
            assert_eq!(out.deltas.len(), 4);
            assert!(
                out.hidden.iter().all(|v| v.to_f32().is_finite()),
                "{inject:?}"
            );
        }
    }
}

#[test]
fn deterministic() {
    let m = LoopedTransformer::random(CFG, lc(3, Inject::Adapter), 7).unwrap();
    let m2 = LoopedTransformer::random(CFG, lc(3, Inject::Adapter), 7).unwrap();
    let x = input(9, CFG.d_model, 3);
    let a = m.forward_cpu(&x, 9);
    assert_eq!(a, m.forward_cpu(&x, 9));
    assert_eq!(a, m2.forward_cpu(&x, 9));
}

#[test]
fn one_loop_equals_one_block() {
    let m = LoopedTransformer::random(CFG, lc(1, Inject::None), 11).unwrap();
    let x = input(6, CFG.d_model, 4);
    let looped = m.forward_cpu(&x, 6);
    let single = m.block.forward_cpu(&x, 6);
    assert_eq!(bits(&looped.hidden), bits(&single));
}

#[test]
fn n_loops_equal_unrolled_shared_blocks() {
    let m = LoopedTransformer::random(CFG, lc(3, Inject::None), 12).unwrap();
    let x = input(4, CFG.d_model, 5);
    let mut h = x.clone();
    for _ in 0..3 {
        h = m.block.forward_cpu(&h, 4);
    }
    assert_eq!(bits(&m.forward_cpu(&x, 4).hidden), bits(&h));
}

#[test]
fn add_injection_matches_manual_unroll() {
    let m = LoopedTransformer::random(CFG, lc(2, Inject::Add), 13).unwrap();
    let x = input(3, CFG.d_model, 6);
    let mut h = x.clone();
    for _ in 0..2 {
        let u: Vec<f16> = h
            .iter()
            .zip(&x)
            .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();
        h = m.block.forward_cpu(&u, 3);
    }
    assert_eq!(bits(&m.forward_cpu(&x, 3).hidden), bits(&h));
}

#[test]
fn halting_respects_min_loops_and_tolerance() {
    let x = input(5, CFG.d_model, 8);
    let halt = |tol, min_loops| LoopConfig {
        loops: 6,
        inject: Inject::Add,
        halting: Some(Halting { tol, min_loops }),
    };
    // Huge tolerance: stops exactly at min_loops.
    let m = LoopedTransformer::random(CFG, halt(1e9, 2), 9).unwrap();
    assert_eq!(m.forward_cpu(&x, 5).loops_run, 2);
    // Zero tolerance: never halts early.
    let m = LoopedTransformer::random(CFG, halt(0.0, 1), 9).unwrap();
    let full = m.forward_cpu(&x, 5);
    assert_eq!(full.loops_run, 6);
    // Halting output equals the non-halting output truncated at the same depth.
    let m3 = LoopedTransformer::random(CFG, lc(2, Inject::Add), 9).unwrap();
    let m_h = LoopedTransformer::random(CFG, halt(1e9, 2), 9).unwrap();
    assert_eq!(m3.forward_cpu(&x, 5).hidden, m_h.forward_cpu(&x, 5).hidden);
    assert_eq!(&full.deltas[..2], &m3.forward_cpu(&x, 5).deltas[..]);
}

#[test]
fn shared_weights_are_one_block() {
    let m = LoopedTransformer::random(CFG, lc(8, Inject::None), 1).unwrap();
    assert_eq!(m.unique_weight_bytes(), m.block.weight_bytes());
    assert_eq!(m.unrolled_weight_bytes(), 8 * m.unique_weight_bytes());
}

#[test]
fn config_validation() {
    let bad = BlockConfig {
        d_model: 96,
        n_heads: 4,
        d_ff: 256,
    };
    assert!(TernaryBlock::random(bad, 0).is_err());
    let bad_heads = BlockConfig {
        d_model: 128,
        n_heads: 3,
        d_ff: 256,
    };
    assert!(TernaryBlock::random(bad_heads, 0).is_err());
    assert!(LoopedTransformer::random(CFG, lc(0, Inject::None), 0).is_err());
    let blk = TernaryBlock::random(CFG, 0).unwrap();
    assert!(LoopedTransformer::new(blk, None, lc(2, Inject::Adapter)).is_err());
    let m = LoopedTransformer::random(CFG, lc(1, Inject::None), 0).unwrap();
    let mut be = CpuBackend::new(&m.block, None);
    assert!(m.forward(&mut be, &input(2, 64, 0), 2).is_err());
}

/// A backend built directly on the scalar reference GEMM: the CPU backend must match it bit for bit.
struct RefBackend<'a> {
    w: Vec<&'a ternary_gemm::PackedTernary>,
}
impl Backend for RefBackend<'_> {
    type Acts = QuantizedActs;
    fn quantize(
        &mut self,
        x: &[f16],
        tokens: usize,
        k: usize,
        gamma: Option<&[f16]>,
    ) -> Result<QuantizedActs, String> {
        Ok(reference::quantize_acts(x, tokens, k, gamma, RMS_EPS))
    }
    fn gemm(
        &mut self,
        w: WeightId,
        a: &QuantizedActs,
        residual: Option<&[f16]>,
    ) -> Result<Vec<f16>, String> {
        Ok(reference::gemm_i8(self.w[w as usize], a, residual))
    }
}

#[test]
fn cpu_backend_matches_reference_gemm() {
    let m = LoopedTransformer::random(CFG, lc(3, Inject::Adapter), 21).unwrap();
    let mut w: Vec<_> = m.block.weights().to_vec();
    w.push(m.adapter.as_ref().unwrap());
    let x = input(7, CFG.d_model, 22);
    let want = m.forward(&mut RefBackend { w }, &x, 7).unwrap();
    let mut cpu = CpuBackend::new(&m.block, m.adapter.as_ref()).with_threads(3);
    assert_eq!(m.forward(&mut cpu, &x, 7).unwrap(), want);
}
