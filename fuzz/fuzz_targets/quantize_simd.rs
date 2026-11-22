//! SIMD activation quantization is bit-identical to the scalar oracle for any finite input.
#![no_main]
use libfuzzer_sys::fuzz_target;
use ternary_gemm::mlp::{quantize_into, quantize_scalar};

fuzz_target!(|data: &[u8]| {
    let x: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|v| v.is_finite())
        .collect();
    let (mut a, mut b) = (vec![0i8; x.len()], vec![0i8; x.len()]);
    let ra = quantize_into(&x, &mut a);
    let rb = quantize_scalar(&x, &mut b);
    assert_eq!((ra.0.to_bits(), ra.1), (rb.0.to_bits(), rb.1));
    assert_eq!(a, b);
});
