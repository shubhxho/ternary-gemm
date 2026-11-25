engine: dense f32 [64, 256, 256, 1] (avx2+fma dots if available)
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 4197.3 ns/sample, 0.24 M samples/s (best of 3)
engine: dense f32 [64, 256, 256, 1] k-major [avx512f k-major fma]
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 3250.0 ns/sample, 0.31 M samples/s (best of 3)
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 1642.6 ns/sample, 0.61 M samples/s (best of 3)
engine: ternary W2A8 [64, 256, 256, 1] CpuEngine [avx512-vnni-vbmi packed2]
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 1061.9 ns/sample, 0.94 M samples/s (best of 3)
engine: ternary weights as 1-byte codes [64, 256, 256, 1] CpuEngine::without_packed [avx512-vnni] (int8-weight cost proxy)
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 1828.3 ns/sample, 0.55 M samples/s (best of 3)
engine: ternary W2A8 (QAT) rerun at 11:54 IST after a noisy first run
inference-only, batch 1 back to back over 6971855 replayed feature vectors: 1051.6 ns/sample, 0.95 M samples/s (best of 3)
