//! safetensors (BitNet b1.58 / any float checkpoint) -> `.tgm` converter (feature `convert`).
//!
//! Linear-layer weights whose names match [`ConvertOptions::ternary_patterns`] are quantized with
//! BitNet's absmean rule ([`crate::pack::quantize_absmean`]) and packed to 2 bits; 1-D tensors
//! (norms, biases) are kept as f32. Other 2-D tensors (embeddings, lm_head) are skipped unless
//! [`ConvertOptions::keep_other_2d`] is set. Use the bf16/f32 release of a model (e.g.
//! `microsoft/bitnet-b1.58-2B-4T-bf16`), not the pre-packed uint8 one.

use crate::format::Tensor;
use crate::pack::{quantize_absmean, PackedTernary, K_ALIGN};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

/// Which tensors become ternary and how they are scaled.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Substrings selecting 2-D weights to ternarize.
    pub ternary_patterns: Vec<String>,
    /// One scale per output row instead of BitNet's single per-tensor scale.
    pub per_row: bool,
    /// Store unmatched 2-D tensors as f32 (they can be large).
    pub keep_other_2d: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            ternary_patterns: [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            .map(String::from)
            .to_vec(),
            per_row: false,
            keep_other_2d: false,
        }
    }
}

/// What happened to each tensor.
#[derive(Debug, Default, Clone)]
pub struct ConvertReport {
    /// (name, n, k, sparsity)
    pub ternarized: Vec<(String, usize, usize, f64)>,
    pub kept_f32: Vec<String>,
    /// (name, reason)
    pub skipped: Vec<(String, String)>,
    pub input_bytes: usize,
    pub packed_bytes: usize,
}

fn to_f32(dtype: Dtype, data: &[u8]) -> Option<Vec<f32>> {
    Some(match dtype {
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        Dtype::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        _ => return None,
    })
}

/// Convert an in-memory safetensors file.
pub fn convert_safetensors(
    bytes: &[u8],
    opts: &ConvertOptions,
) -> Result<(BTreeMap<String, Tensor>, ConvertReport), String> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| format!("safetensors: {e}"))?;
    let mut out = BTreeMap::new();
    let mut rep = ConvertReport {
        input_bytes: bytes.len(),
        ..Default::default()
    };
    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    for name in names {
        let t = st.tensor(&name).map_err(|e| format!("{name}: {e}"))?;
        let shape = t.shape().to_vec();
        let Some(data) = to_f32(t.dtype(), t.data()) else {
            rep.skipped
                .push((name, format!("dtype {:?} not supported", t.dtype())));
            continue;
        };
        let matches = opts
            .ternary_patterns
            .iter()
            .any(|p| name.contains(p.as_str()));
        match shape.as_slice() {
            [n, k] if matches => {
                let (n, k) = (*n, *k);
                if k % K_ALIGN != 0 {
                    rep.skipped
                        .push((name, format!("K={k} is not a multiple of {K_ALIGN}")));
                    continue;
                }
                let (w, scales) = quantize_absmean(&data, n, k, opts.per_row);
                let p = PackedTernary::from_ternary(&w, n, k, scales)
                    .map_err(|e| format!("{name}: {e}"))?;
                rep.packed_bytes += p.weight_bytes() + 4 * n;
                rep.ternarized.push((name.clone(), n, k, p.sparsity()));
                out.insert(name, Tensor::Ternary(p));
            }
            [_] => {
                rep.packed_bytes += 4 * data.len();
                rep.kept_f32.push(name.clone());
                out.insert(name, Tensor::F32(data));
            }
            [_, _] if opts.keep_other_2d => {
                rep.packed_bytes += 4 * data.len();
                rep.kept_f32.push(name.clone());
                out.insert(name, Tensor::F32(data));
            }
            _ => rep
                .skipped
                .push((name, format!("shape {shape:?} not converted"))),
        }
    }
    out.insert(
        "__meta__".into(),
        Tensor::Text(format!(
            "{{\"converter\":\"ternary-gemm {}\",\"quantization\":\"absmean\",\"per_row\":{}}}",
            env!("CARGO_PKG_VERSION"),
            opts.per_row
        )),
    );
    Ok((out, rep))
}
