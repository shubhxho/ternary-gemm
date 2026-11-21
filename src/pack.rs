//! Ternary weight quantization and 2-bit packing.
//!
//! Codes are `w + 1` in `{0, 1, 2}`; 16 codes per `u32`. Two layouts are produced:
//!
//! * **permuted** (used by the fast kernels): inside a word, weight `16*kb + 4*j + i` sits at bits
//!   `[8*i + 2*j, +2)`, so `(word >> 2*j) & 0x03030303` is four consecutive weights as bytes, ready
//!   for `dp4a`. Each row is split into chunks of [`CHUNK_WORDS`] words; in a chunk of `wc` words
//!   with `l = wc / 4` lanes, the word for k-block `q*l + lane` is stored at chunk offset `4*lane + q`.
//! * **natural** (used by the naive baseline kernel): weight `k` at word `k / 16`, bits `2*(k % 16)`.

/// Words per chunk processed by one warp iteration (32 lanes x one `uint4`).
pub const CHUNK_WORDS: usize = 128;

/// `K` must be a multiple of this (one lane's `uint4` covers 64 weights).
pub const K_ALIGN: usize = 64;

/// Errors from building packed weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// `K` is not a multiple of [`K_ALIGN`].
    KNotAligned(usize),
    /// Data length does not match `n * k`.
    BadLength { expected: usize, got: usize },
    /// A weight was not in `{-1, 0, 1}`.
    NotTernary { index: usize, value: i8 },
    /// Wrong number of scales.
    BadScales { expected: usize, got: usize },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackError::KNotAligned(k) => write!(f, "K={k} must be a multiple of {K_ALIGN}"),
            PackError::BadLength { expected, got } => {
                write!(f, "expected {expected} weights, got {got}")
            }
            PackError::NotTernary { index, value } => {
                write!(f, "weight[{index}] = {value} is not in {{-1,0,1}}")
            }
            PackError::BadScales { expected, got } => {
                write!(f, "expected {expected} row scales, got {got}")
            }
        }
    }
}
impl std::error::Error for PackError {}

/// A ternary weight matrix `[n][k]` with one f32 scale per output row.
#[derive(Debug, Clone)]
pub struct PackedTernary {
    pub n: usize,
    pub k: usize,
    /// Permuted layout, `n * k / 16` words.
    pub permuted: Vec<u32>,
    /// Natural layout, `n * k / 16` words.
    pub natural: Vec<u32>,
    /// Per-row scale (BitNet's per-tensor scale is the special case where all rows are equal).
    pub row_scale: Vec<f32>,
}

/// Position of k-block `kb` (16 weights) of a row inside the permuted row, in words.
#[inline]
pub fn permuted_word_index(kb: usize, kw: usize) -> usize {
    let base = (kb / CHUNK_WORDS) * CHUNK_WORDS;
    let wc = CHUNK_WORDS.min(kw - base);
    let l = wc / 4;
    let off = kb - base; // = q * l + lane
    let q = off / l;
    let lane = off % l;
    base + 4 * lane + q
}

/// Pack 16 codes (weights k0..k0+16) into one permuted word.
#[inline]
fn pack_word_permuted(codes: &[u8]) -> u32 {
    let mut w = 0u32;
    for (t, &c) in codes.iter().enumerate() {
        let (j, i) = (t / 4, t % 4);
        w |= (c as u32) << (8 * i + 2 * j);
    }
    w
}

#[inline]
fn pack_word_natural(codes: &[u8]) -> u32 {
    let mut w = 0u32;
    for (t, &c) in codes.iter().enumerate() {
        w |= (c as u32) << (2 * t);
    }
    w
}

impl PackedTernary {
    /// Pack a row-major `[n][k]` matrix of `{-1,0,1}` with per-row scales.
    pub fn from_ternary(
        w: &[i8],
        n: usize,
        k: usize,
        row_scale: Vec<f32>,
    ) -> Result<Self, PackError> {
        if !k.is_multiple_of(K_ALIGN) || k == 0 {
            return Err(PackError::KNotAligned(k));
        }
        if w.len() != n * k {
            return Err(PackError::BadLength {
                expected: n * k,
                got: w.len(),
            });
        }
        if row_scale.len() != n {
            return Err(PackError::BadScales {
                expected: n,
                got: row_scale.len(),
            });
        }
        let kw = k / 16;
        let mut permuted = vec![0u32; n * kw];
        let mut natural = vec![0u32; n * kw];
        let mut codes = [0u8; 16];
        for row in 0..n {
            for kb in 0..kw {
                for t in 0..16 {
                    let idx = row * k + kb * 16 + t;
                    let v = w[idx];
                    if !(-1..=1).contains(&v) {
                        return Err(PackError::NotTernary {
                            index: idx,
                            value: v,
                        });
                    }
                    codes[t] = (v + 1) as u8;
                }
                permuted[row * kw + permuted_word_index(kb, kw)] = pack_word_permuted(&codes);
                natural[row * kw + kb] = pack_word_natural(&codes);
            }
        }
        Ok(Self {
            n,
            k,
            permuted,
            natural,
            row_scale,
        })
    }

    /// Unpack weight `(row, k)` from the permuted layout (used by the CPU reference).
    #[inline]
    pub fn get(&self, row: usize, k: usize) -> i8 {
        let kw = self.k / 16;
        let kb = k / 16;
        let t = k % 16;
        let (j, i) = (t / 4, t % 4);
        let word = self.permuted[row * kw + permuted_word_index(kb, kw)];
        ((word >> (8 * i + 2 * j)) & 3) as i8 - 1
    }

    /// Unpack weight `(row, k)` from the natural layout.
    #[inline]
    pub fn get_natural(&self, row: usize, k: usize) -> i8 {
        let word = self.natural[row * (self.k / 16) + k / 16];
        ((word >> (2 * (k % 16))) & 3) as i8 - 1
    }

    /// Bytes of packed weights (one layout).
    pub fn weight_bytes(&self) -> usize {
        self.permuted.len() * 4
    }
}

/// BitNet b1.58 "absmean" weight quantization of an f32 `[n][k]` matrix.
///
/// `per_row = false` reproduces BitNet (one scale for the whole tensor); `true` uses a scale per
/// output row. Returns ternary weights and the per-row dequantization scales.
pub fn quantize_absmean(w: &[f32], n: usize, k: usize, per_row: bool) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), n * k);
    let mean_abs =
        |s: &[f32]| (s.iter().map(|v| v.abs() as f64).sum::<f64>() / s.len() as f64) as f32;
    let mut scales = vec![0f32; n];
    if per_row {
        for r in 0..n {
            scales[r] = mean_abs(&w[r * k..(r + 1) * k]).max(1e-5);
        }
    } else {
        let s = mean_abs(w).max(1e-5);
        scales.iter_mut().for_each(|x| *x = s);
    }
    let mut q = vec![0i8; n * k];
    for r in 0..n {
        for c in 0..k {
            q[r * k + c] = (w[r * k + c] / scales[r]).round().clamp(-1.0, 1.0) as i8;
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_bijection() {
        for &k in &[64usize, 128, 2048, 2560, 4096, 6912, 11008] {
            let kw = k / 16;
            let mut seen = vec![false; kw];
            for kb in 0..kw {
                let p = permuted_word_index(kb, kw);
                assert!(!seen[p], "k={k} kb={kb} collides at {p}");
                seen[p] = true;
            }
        }
    }

    #[test]
    fn roundtrip() {
        let (n, k) = (7, 2560 + 64);
        let w: Vec<i8> = (0..n * k)
            .map(|i| ((i * 2654435761usize) >> 7) as i8 % 2)
            .map(|v| v.signum())
            .collect();
        let p = PackedTernary::from_ternary(&w, n, k, vec![1.0; n]).unwrap();
        for r in 0..n {
            for c in 0..k {
                assert_eq!(p.get(r, c), w[r * k + c]);
                assert_eq!(p.get_natural(r, c), w[r * k + c]);
            }
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            PackedTernary::from_ternary(&[0; 32], 1, 32, vec![1.0]).unwrap_err(),
            PackError::KNotAligned(32)
        );
        let mut w = vec![0i8; 64];
        w[5] = 2;
        assert!(matches!(
            PackedTernary::from_ternary(&w, 1, 64, vec![1.0]),
            Err(PackError::NotTernary { index: 5, .. })
        ));
    }
}
