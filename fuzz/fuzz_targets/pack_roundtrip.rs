//! pack -> unpack is the identity; natural and permuted layouts agree; from_natural inverts it.
#![no_main]
use libfuzzer_sys::fuzz_target;
use ternary_gemm::pack::PackedTernary;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let n = (data[0] % 4) as usize + 1;
    let k = 64 * ((data[1] % 40) as usize + 1);
    let w: Vec<i8> = (0..n * k).map(|i| (data[2 + i % (data.len() - 2)] % 3) as i8 - 1).collect();
    let p = PackedTernary::from_ternary(&w, n, k, vec![1.0; n]).unwrap();
    assert_eq!(p.unpack(), w);
    let q = PackedTernary::from_natural(p.natural.clone(), n, k, p.row_scale.clone()).unwrap();
    assert_eq!(q.permuted, p.permuted);
});
