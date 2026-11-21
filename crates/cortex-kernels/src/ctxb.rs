//! `.ctxb` compact binary model format (Phase 1).
//!
//! Layout, all little-endian:
//! ```text
//! Header (64 bytes):
//!   magic[4] = "CTXB", version u32 = 1, d_model u32, num_macros u32,
//!   num_heads u32, head_dim u32, vocab_size u32, num_tensors u32,
//!   payload_len u64, crc32 u32 (IEEE, over the payload only),
//!   reserved[16] = 0
//! Payload: for each tensor:
//!   kind u8 (0 = BitLinear, 1 = F32Vector), name_len u16, name bytes,
//!   rows u32, cols u32, scales f32[rows] (BitLinear only),
//!   data_len u64, data bytes
//!   (BitLinear: data = Tile16x4 bytes from `tile_bytes()`;
//!    F32Vector: data = f32[rows] little-endian, cols must be 1)
//! ```
//!
//! The Python exporter (`tools/train_seed_qat.py`) writes exactly
//! this layout, and `CtxbReader` verifies magic, version and CRC32.

use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::bit158::BitLinearWeights;

pub const CTXB_MAGIC: &[u8; 4] = b"CTXB";
pub const CTXB_VERSION: u32 = 1;
pub const CTXB_HEADER_LEN: usize = 64;

/// CRC32 (IEEE, poly 0xEDB88320), table built by a `const fn`.
pub const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc32_table();

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = CRC_TABLE[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CtxbHeader {
    pub d_model: u32,
    pub num_macros: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub num_tensors: u32,
    /// FFN hidden dim (0 in old files = derive as `2 * d_model`).
    pub ffn_hidden: u32,
    /// Hippocampus hidden dim (0 in old files = derive as `2 * d_model`).
    pub hippo_hidden: u32,
}

impl CtxbHeader {
    pub fn new(
        d_model: u32,
        num_macros: u32,
        num_heads: u32,
        head_dim: u32,
        vocab_size: u32,
    ) -> Self {
        Self {
            d_model,
            num_macros,
            num_heads,
            head_dim,
            vocab_size,
            num_tensors: 0,
            ffn_hidden: 0,
            hippo_hidden: 0,
        }
    }

    pub fn ffn_hidden_or_default(&self) -> u32 {
        if self.ffn_hidden == 0 {
            self.d_model * 2
        } else {
            self.ffn_hidden
        }
    }
    pub fn hippo_hidden_or_default(&self) -> u32 {
        if self.hippo_hidden == 0 {
            self.d_model * 2
        } else {
            self.hippo_hidden
        }
    }
}

#[derive(Clone, Debug)]
pub enum CtxbTensorKind {
    BitLinear(BitLinearWeights),
    F32Vector(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct CtxbTensor {
    pub name: String,
    pub kind: CtxbTensorKind,
}

impl CtxbTensor {
    pub fn bitlinear(name: impl Into<String>, w: BitLinearWeights) -> Self {
        Self {
            name: name.into(),
            kind: CtxbTensorKind::BitLinear(w),
        }
    }
    pub fn vector(name: impl Into<String>, v: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            kind: CtxbTensorKind::F32Vector(v),
        }
    }
}

#[derive(Debug)]
pub enum CtxbError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    BadCrc { expected: u32, actual: u32 },
    Truncated,
    Invalid(String),
}

impl fmt::Display for CtxbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CtxbError::Io(e) => write!(f, "ctxb io error: {e}"),
            CtxbError::BadMagic => write!(f, "ctxb bad magic (not CTXB)"),
            CtxbError::UnsupportedVersion(v) => write!(f, "ctxb unsupported version {v}"),
            CtxbError::BadCrc { expected, actual } => {
                write!(
                    f,
                    "ctxb CRC32 mismatch: expected {expected:#x}, got {actual:#x}"
                )
            }
            CtxbError::Truncated => write!(f, "ctxb file truncated"),
            CtxbError::Invalid(s) => write!(f, "ctxb invalid: {s}"),
        }
    }
}

impl std::error::Error for CtxbError {}
impl From<std::io::Error> for CtxbError {
    fn from(e: std::io::Error) -> Self {
        CtxbError::Io(e)
    }
}

fn read_exact_or_truncated<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), CtxbError> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            CtxbError::Truncated
        } else {
            CtxbError::Io(e)
        }
    })
}

/// Write a `.ctxb` file. Returns the number of payload bytes written.
pub fn write_ctxb(
    path: &Path,
    header: &CtxbHeader,
    tensors: &[CtxbTensor],
) -> Result<u64, CtxbError> {
    let mut payload: Vec<u8> = Vec::new();
    for t in tensors {
        let name = t.name.as_bytes();
        if name.len() > u16::MAX as usize {
            return Err(CtxbError::Invalid("tensor name too long".into()));
        }
        match &t.kind {
            CtxbTensorKind::BitLinear(w) => {
                payload.push(0u8);
                payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
                payload.extend_from_slice(name);
                payload.extend_from_slice(&(w.rows as u32).to_le_bytes());
                payload.extend_from_slice(&(w.cols as u32).to_le_bytes());
                for s in &w.scales {
                    payload.extend_from_slice(&s.to_le_bytes());
                }
                let tiles = w.tile_bytes();
                payload.extend_from_slice(&(tiles.len() as u64).to_le_bytes());
                payload.extend_from_slice(&tiles);
            }
            CtxbTensorKind::F32Vector(v) => {
                payload.push(1u8);
                payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
                payload.extend_from_slice(name);
                payload.extend_from_slice(&(v.len() as u32).to_le_bytes());
                payload.extend_from_slice(&1u32.to_le_bytes());
                let data_len = (v.len() * 4) as u64;
                payload.extend_from_slice(&data_len.to_le_bytes());
                for x in v {
                    payload.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    let crc = crc32(&payload);
    let mut hdr = [0u8; CTXB_HEADER_LEN];
    hdr[0..4].copy_from_slice(CTXB_MAGIC);
    hdr[4..8].copy_from_slice(&CTXB_VERSION.to_le_bytes());
    hdr[8..12].copy_from_slice(&header.d_model.to_le_bytes());
    hdr[12..16].copy_from_slice(&header.num_macros.to_le_bytes());
    hdr[16..20].copy_from_slice(&header.num_heads.to_le_bytes());
    hdr[20..24].copy_from_slice(&header.head_dim.to_le_bytes());
    hdr[24..28].copy_from_slice(&header.vocab_size.to_le_bytes());
    hdr[28..32].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
    hdr[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    hdr[40..44].copy_from_slice(&crc.to_le_bytes());
    hdr[44..48].copy_from_slice(&header.ffn_hidden.to_le_bytes());
    hdr[48..52].copy_from_slice(&header.hippo_hidden.to_le_bytes());
    // bytes 52..64 reserved = 0
    let mut f = File::create(path)?;
    f.write_all(&hdr)?;
    f.write_all(&payload)?;
    Ok(payload.len() as u64)
}

/// Read and verify a `.ctxb` file.
pub fn read_ctxb(path: &Path) -> Result<(CtxbHeader, Vec<CtxbTensor>), CtxbError> {
    let mut f = File::open(path)?;
    let mut hdr = [0u8; CTXB_HEADER_LEN];
    read_exact_or_truncated(&mut f, &mut hdr)?;
    if &hdr[0..4] != CTXB_MAGIC {
        return Err(CtxbError::BadMagic);
    }
    let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    if version != CTXB_VERSION {
        return Err(CtxbError::UnsupportedVersion(version));
    }
    let header = CtxbHeader {
        d_model: u32::from_le_bytes(hdr[8..12].try_into().unwrap()),
        num_macros: u32::from_le_bytes(hdr[12..16].try_into().unwrap()),
        num_heads: u32::from_le_bytes(hdr[16..20].try_into().unwrap()),
        head_dim: u32::from_le_bytes(hdr[20..24].try_into().unwrap()),
        vocab_size: u32::from_le_bytes(hdr[24..28].try_into().unwrap()),
        num_tensors: u32::from_le_bytes(hdr[28..32].try_into().unwrap()),
        ffn_hidden: u32::from_le_bytes(hdr[44..48].try_into().unwrap()),
        hippo_hidden: u32::from_le_bytes(hdr[48..52].try_into().unwrap()),
    };
    let payload_len = u64::from_le_bytes(hdr[32..40].try_into().unwrap()) as usize;
    let expected_crc = u32::from_le_bytes(hdr[40..44].try_into().unwrap());
    let mut payload = vec![0u8; payload_len];
    read_exact_or_truncated(&mut f, &mut payload)?;
    let actual_crc = crc32(&payload);
    if actual_crc != expected_crc {
        return Err(CtxbError::BadCrc {
            expected: expected_crc,
            actual: actual_crc,
        });
    }
    let mut tensors = Vec::with_capacity(header.num_tensors as usize);
    let mut off = 0usize;
    let need = |off: usize, n: usize| -> Result<(), CtxbError> {
        if off + n > payload.len() {
            Err(CtxbError::Truncated)
        } else {
            Ok(())
        }
    };
    for _ in 0..header.num_tensors {
        need(off, 1)?;
        let kind = payload[off];
        off += 1;
        need(off, 2)?;
        let name_len = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        need(off, name_len)?;
        let name = String::from_utf8(payload[off..off + name_len].to_vec())
            .map_err(|_| CtxbError::Invalid("tensor name is not UTF-8".into()))?;
        off += name_len;
        need(off, 8)?;
        let rows = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        let cols = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        match kind {
            0 => {
                need(off, rows * 4)?;
                let mut scales = Vec::with_capacity(rows);
                for i in 0..rows {
                    scales.push(f32::from_le_bytes(
                        payload[off + i * 4..off + i * 4 + 4].try_into().unwrap(),
                    ));
                }
                off += rows * 4;
                need(off, 8)?;
                let data_len =
                    u64::from_le_bytes(payload[off..off + 8].try_into().unwrap()) as usize;
                off += 8;
                need(off, data_len)?;
                let w = BitLinearWeights::from_tile_bytes(
                    &payload[off..off + data_len],
                    rows,
                    cols,
                    &scales,
                );
                off += data_len;
                tensors.push(CtxbTensor::bitlinear(name, w));
            }
            1 => {
                if cols != 1 {
                    return Err(CtxbError::Invalid(
                        "F32Vector tensor must have cols == 1".into(),
                    ));
                }
                need(off, 8)?;
                let data_len =
                    u64::from_le_bytes(payload[off..off + 8].try_into().unwrap()) as usize;
                off += 8;
                if data_len != rows * 4 {
                    return Err(CtxbError::Invalid("F32Vector data length mismatch".into()));
                }
                need(off, data_len)?;
                let mut v = Vec::with_capacity(rows);
                for i in 0..rows {
                    v.push(f32::from_le_bytes(
                        payload[off + i * 4..off + i * 4 + 4].try_into().unwrap(),
                    ));
                }
                off += data_len;
                tensors.push(CtxbTensor::vector(name, v));
            }
            other => return Err(CtxbError::Invalid(format!("unknown tensor kind {other}"))),
        }
    }
    Ok((header, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta_net::xorshift64star;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn ctxb_roundtrip() {
        let mut st = 42u64;
        let w: Vec<i8> = (0..37 * 40)
            .map(|_| (xorshift64star(&mut st) % 3) as i8 - 1)
            .collect();
        let m = BitLinearWeights::from_ternary_matrix(&w, 37, 40, &[0.2; 37]);
        let tensors = vec![
            CtxbTensor::bitlinear("layer0.q_proj", m.clone()),
            CtxbTensor::vector("layer0.rms", vec![1.0, 0.5, 2.0]),
        ];
        let header = CtxbHeader::new(40, 1, 2, 20, 256);
        let dir = std::env::temp_dir();
        let path = dir.join("cortex-test-roundtrip.ctxb");
        write_ctxb(&path, &header, &tensors).unwrap();
        let (h2, t2) = read_ctxb(&path).unwrap();
        assert_eq!(h2.d_model, 40);
        assert_eq!(h2.num_tensors, 2);
        assert_eq!(t2.len(), 2);
        assert_eq!(t2[0].name, "layer0.q_proj");
        if let CtxbTensorKind::BitLinear(w2) = &t2[0].kind {
            assert_eq!(w2.tile_bytes(), m.tile_bytes());
            assert_eq!(w2.scales, m.scales);
        } else {
            panic!("expected BitLinear");
        }
        if let CtxbTensorKind::F32Vector(v) = &t2[1].kind {
            assert_eq!(v, &vec![1.0, 0.5, 2.0]);
        } else {
            panic!("expected F32Vector");
        }
        // Corrupt one payload byte -> CRC must fail.
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(read_ctxb(&path), Err(CtxbError::BadCrc { .. })));
        let _ = std::fs::remove_file(&path);
    }
}
