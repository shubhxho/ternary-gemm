//! `.tgm` container round trips and rejects corrupted input.

use std::collections::BTreeMap;
use ternary_gemm::format::{from_bytes, to_bytes, FormatError, Tensor};
use ternary_gemm::pack::PackedTernary;

fn sample() -> BTreeMap<String, Tensor> {
    let w: Vec<i8> = (0..3 * 128).map(|i| ((i * 7) % 3) as i8 - 1).collect();
    let p = PackedTernary::from_ternary(&w, 3, 128, vec![0.5, 0.25, 2.0]).unwrap();
    let mut t = BTreeMap::new();
    t.insert("layers.0.weight".to_string(), Tensor::Ternary(p));
    t.insert(
        "layers.0.bias".to_string(),
        Tensor::F32(vec![1.0, -2.5, 3.25]),
    );
    t.insert(
        "meta".to_string(),
        Tensor::Text("{\"arch\":\"test\"}".to_string()),
    );
    t
}

#[test]
fn roundtrip() {
    let t = sample();
    let b = to_bytes(&t).unwrap();
    let back = from_bytes(&b).unwrap();
    assert_eq!(back, t);
    // the permuted layout is rebuilt identically
    if let (Tensor::Ternary(a), Tensor::Ternary(b)) =
        (&t["layers.0.weight"], &back["layers.0.weight"])
    {
        assert_eq!(a.permuted, b.permuted);
    }
}

#[test]
fn detects_corruption_and_truncation() {
    let b = to_bytes(&sample()).unwrap();
    for i in [0usize, 7, 20, b.len() / 2, b.len() - 1] {
        let mut c = b.clone();
        c[i] ^= 0x40;
        assert!(from_bytes(&c).is_err(), "flip at {i} accepted");
    }
    for len in [0usize, 5, 12, b.len() - 9, b.len() - 1] {
        assert!(
            from_bytes(&b[..len]).is_err(),
            "truncation to {len} accepted"
        );
    }
    assert!(matches!(
        from_bytes(b"NOTATGMFILE.........."),
        Err(FormatError::BadMagic)
    ));
}

#[test]
fn file_save_load() {
    let dir = std::env::temp_dir().join(format!("tgm-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("m.tgm");
    ternary_gemm::format::save(&p, &sample()).unwrap();
    assert_eq!(ternary_gemm::format::load(&p).unwrap(), sample());
    std::fs::remove_dir_all(&dir).unwrap();
}
