//! `.tgm` container: packed ternary matrices, f32 vectors and small text metadata in one
//! little-endian file, so a converted model loads without re-quantizing.
//!
//! Layout (version 1), all integers little-endian:
//! ```text
//! magic  "TGEMM\0" + u16 version (1)
//! u32    tensor count
//! per tensor:
//!   u16 name length, name (UTF-8)
//!   u8  kind: 0 = ternary matrix, 1 = f32 vector, 2 = UTF-8 text
//!   kind 0: u32 n, u32 k, f32 row_scale[n], u32 natural_words[n * k / 16]
//!   kind 1: u32 len, f32 data[len]
//!   kind 2: u32 len, bytes[len]
//! u64    FNV-1a hash of every preceding byte
//! ```
//! The natural layout is the canonical on-disk form; the GPU's permuted layout is rebuilt on load.
//! See `docs/FORMAT.md`.

use crate::pack::{PackError, PackedTernary};
use std::collections::BTreeMap;

const MAGIC: &[u8; 6] = b"TGEMM\0";
const VERSION: u16 = 1;

/// One entry of a `.tgm` file.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    Ternary(PackedTernary),
    F32(Vec<f32>),
    Text(String),
}

/// Errors reading or writing `.tgm` files.
#[derive(Debug)]
pub enum FormatError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u16),
    Truncated,
    Checksum,
    BadName,
    BadKind(u8),
    Pack(PackError),
    TooLarge(String),
}
impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Io(e) => write!(f, "I/O error: {e}"),
            FormatError::BadMagic => write!(f, "not a .tgm file (bad magic)"),
            FormatError::UnsupportedVersion(v) => write!(f, "unsupported .tgm version {v}"),
            FormatError::Truncated => write!(f, "file is truncated"),
            FormatError::Checksum => write!(f, "checksum mismatch (corrupted file)"),
            FormatError::BadName => write!(f, "tensor name is not valid UTF-8"),
            FormatError::BadKind(k) => write!(f, "unknown tensor kind {k}"),
            FormatError::Pack(e) => write!(f, "invalid ternary tensor: {e}"),
            FormatError::TooLarge(s) => write!(f, "too large: {s}"),
        }
    }
}
impl std::error::Error for FormatError {}
impl From<std::io::Error> for FormatError {
    fn from(e: std::io::Error) -> Self {
        FormatError::Io(e)
    }
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Serialize tensors (sorted by name) to bytes.
pub fn to_bytes(tensors: &BTreeMap<String, Tensor>) -> Result<Vec<u8>, FormatError> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    o.extend_from_slice(&VERSION.to_le_bytes());
    o.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    let u32_of = |v: usize, what: &str| {
        u32::try_from(v).map_err(|_| FormatError::TooLarge(format!("{what} = {v}")))
    };
    for (name, t) in tensors {
        let nb = name.as_bytes();
        let nl =
            u16::try_from(nb.len()).map_err(|_| FormatError::TooLarge(format!("name {name}")))?;
        o.extend_from_slice(&nl.to_le_bytes());
        o.extend_from_slice(nb);
        match t {
            Tensor::Ternary(p) => {
                o.push(0);
                o.extend_from_slice(&u32_of(p.n, "n")?.to_le_bytes());
                o.extend_from_slice(&u32_of(p.k, "k")?.to_le_bytes());
                for s in &p.row_scale {
                    o.extend_from_slice(&s.to_le_bytes());
                }
                for w in &p.natural {
                    o.extend_from_slice(&w.to_le_bytes());
                }
            }
            Tensor::F32(v) => {
                o.push(1);
                o.extend_from_slice(&u32_of(v.len(), "len")?.to_le_bytes());
                for x in v {
                    o.extend_from_slice(&x.to_le_bytes());
                }
            }
            Tensor::Text(s) => {
                o.push(2);
                o.extend_from_slice(&u32_of(s.len(), "len")?.to_le_bytes());
                o.extend_from_slice(s.as_bytes());
            }
        }
    }
    let h = fnv1a(&o);
    o.extend_from_slice(&h.to_le_bytes());
    Ok(o)
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        let end = self.at.checked_add(n).ok_or(FormatError::Truncated)?;
        let s = self.b.get(self.at..end).ok_or(FormatError::Truncated)?;
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, FormatError> {
        let raw = self.take(n.checked_mul(4).ok_or(FormatError::Truncated)?)?;
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

/// Parse a `.tgm` image. Never panics on malformed input (fuzzed in `fuzz/`).
pub fn from_bytes(bytes: &[u8]) -> Result<BTreeMap<String, Tensor>, FormatError> {
    if bytes.len() < MAGIC.len() + 2 + 4 + 8 {
        return Err(if bytes.starts_with(MAGIC) {
            FormatError::Truncated
        } else {
            FormatError::BadMagic
        });
    }
    if &bytes[..6] != MAGIC {
        return Err(FormatError::BadMagic);
    }
    let (body, tail) = bytes.split_at(bytes.len() - 8);
    if fnv1a(body) != u64::from_le_bytes(tail.try_into().unwrap()) {
        return Err(FormatError::Checksum);
    }
    let mut r = Reader { b: body, at: 6 };
    let v = r.u16()?;
    if v != VERSION {
        return Err(FormatError::UnsupportedVersion(v));
    }
    let count = r.u32()?;
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let nl = r.u16()? as usize;
        let name = std::str::from_utf8(r.take(nl)?)
            .map_err(|_| FormatError::BadName)?
            .to_string();
        let t = match r.u8()? {
            0 => {
                let n = r.u32()? as usize;
                let k = r.u32()? as usize;
                let words = n.checked_mul(k).ok_or(FormatError::Truncated)? / 16;
                // Reject before allocating: the data must actually be present.
                if words.saturating_mul(4).saturating_add(n.saturating_mul(4)) > body.len() - r.at {
                    return Err(FormatError::Truncated);
                }
                let row_scale = r.f32s(n)?;
                let raw = r.take(words * 4)?;
                let natural: Vec<u32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                Tensor::Ternary(
                    PackedTernary::from_natural(natural, n, k, row_scale)
                        .map_err(FormatError::Pack)?,
                )
            }
            1 => {
                let len = r.u32()? as usize;
                Tensor::F32(r.f32s(len)?)
            }
            2 => {
                let len = r.u32()? as usize;
                let s = std::str::from_utf8(r.take(len)?).map_err(|_| FormatError::BadName)?;
                Tensor::Text(s.to_string())
            }
            k => return Err(FormatError::BadKind(k)),
        };
        out.insert(name, t);
    }
    if r.at != body.len() {
        return Err(FormatError::Truncated);
    }
    Ok(out)
}

/// Write a `.tgm` file.
pub fn save(
    path: impl AsRef<std::path::Path>,
    t: &BTreeMap<String, Tensor>,
) -> Result<(), FormatError> {
    Ok(std::fs::write(path, to_bytes(t)?)?)
}

/// Read a `.tgm` file.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<BTreeMap<String, Tensor>, FormatError> {
    from_bytes(&std::fs::read(path)?)
}
