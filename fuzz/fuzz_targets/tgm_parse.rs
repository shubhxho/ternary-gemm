//! The .tgm parser must never panic or over-allocate on arbitrary bytes, and anything it accepts
//! must re-serialize to the same bytes.
#![no_main]
use libfuzzer_sys::fuzz_target;
use ternary_gemm::format::{from_bytes, to_bytes};

fuzz_target!(|data: &[u8]| {
    if let Ok(t) = from_bytes(data) {
        assert_eq!(to_bytes(&t).unwrap(), data);
    }
});
