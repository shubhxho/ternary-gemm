//! safetensors -> .tgm conversion on a synthetic checkpoint.

use half::bf16;
use safetensors::{serialize, tensor::TensorView, Dtype};
use std::collections::HashMap;
use ternary_gemm::convert::{convert_safetensors, ConvertOptions};
use ternary_gemm::format::{from_bytes, to_bytes, Tensor};
use ternary_gemm::pack::quantize_absmean;

#[test]
fn converts_bf16_checkpoint() {
    let (n, k) = (8usize, 128usize);
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let wb: Vec<u8> = w
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect();
    let norm: Vec<u8> = (0..k).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let emb: Vec<u8> = vec![0u8; 4 * 16 * k];
    let odd: Vec<u8> = vec![0u8; 2 * n * 100];
    let mut m = HashMap::new();
    m.insert(
        "model.layers.0.self_attn.q_proj.weight",
        TensorView::new(Dtype::BF16, vec![n, k], &wb).unwrap(),
    );
    m.insert(
        "model.layers.0.input_layernorm.weight",
        TensorView::new(Dtype::F32, vec![k], &norm).unwrap(),
    );
    m.insert(
        "model.embed_tokens.weight",
        TensorView::new(Dtype::F32, vec![16, k], &emb).unwrap(),
    );
    m.insert(
        "model.layers.0.mlp.up_proj.weight",
        TensorView::new(Dtype::BF16, vec![n, 100], &odd).unwrap(),
    );
    let bytes = serialize(m, None).unwrap();

    let (t, rep) = convert_safetensors(&bytes, &ConvertOptions::default()).unwrap();
    assert_eq!(rep.ternarized.len(), 1);
    assert_eq!(
        rep.kept_f32,
        vec!["model.layers.0.input_layernorm.weight".to_string()]
    );
    assert_eq!(rep.skipped.len(), 2, "{:?}", rep.skipped); // embedding + K=100

    let Tensor::Ternary(p) = &t["model.layers.0.self_attn.q_proj.weight"] else {
        panic!()
    };
    let wf: Vec<f32> = w.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
    let (want, scales) = quantize_absmean(&wf, n, k, false);
    assert_eq!(p.unpack(), want);
    assert_eq!(p.row_scale, scales);

    // survives the container round trip
    assert_eq!(from_bytes(&to_bytes(&t).unwrap()).unwrap(), t);
}
