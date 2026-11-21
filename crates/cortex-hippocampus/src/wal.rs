//! SSD append-only WAL for high-surprise episodes (Phase 3).
//!
//! Format: a sequence of length-prefixed records. Each record:
//! ```text
//! len u32 | payload | crc32 u32 (IEEE, over payload)
//! payload = seq u64 | surprise f32 | kind u8
//!         | ctx_len u32 | ctx token ids u32[ctx_len]
//!         | k_len u32 | k f32[k_len] | v_len u32 | v f32[v_len]
//! ```
//! A torn tail (crash mid-write) fails CRC / length checks; the
//! reader stops there and reports `truncated = true`, keeping every
//! complete record before it. The writer flushes to the OS buffer on
//! every record and `fsync`s every `sync_every` records (default 20,
//! ~5 s of heavy writing at the spec's cadence).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use cortex_kernels::crc32;

#[derive(Clone, Debug, PartialEq)]
pub struct WalRecord {
    pub seq: u64,
    pub surprise: f32,
    /// Token ids of the current context window (may be empty).
    pub context: Vec<u32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl WalRecord {
    fn encode(&self) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&self.seq.to_le_bytes());
        p.extend_from_slice(&self.surprise.to_le_bytes());
        p.push(0u8); // kind: 0 = learn pair
        p.extend_from_slice(&(self.context.len() as u32).to_le_bytes());
        for t in &self.context {
            p.extend_from_slice(&t.to_le_bytes());
        }
        p.extend_from_slice(&(self.k.len() as u32).to_le_bytes());
        for x in &self.k {
            p.extend_from_slice(&x.to_le_bytes());
        }
        p.extend_from_slice(&(self.v.len() as u32).to_le_bytes());
        for x in &self.v {
            p.extend_from_slice(&x.to_le_bytes());
        }
        p
    }

    fn decode(payload: &[u8]) -> Option<WalRecord> {
        let mut off = 0usize;
        let need = |off: usize, n: usize| off.checked_add(n).filter(|&e| e <= payload.len());
        let take_u64 = |off: &mut usize| -> Option<u64> {
            let e = need(*off, 8)?;
            let v = u64::from_le_bytes(payload[*off..e].try_into().ok()?);
            *off = e;
            Some(v)
        };
        let take_u32 = |off: &mut usize| -> Option<u32> {
            let e = need(*off, 4)?;
            let v = u32::from_le_bytes(payload[*off..e].try_into().ok()?);
            *off = e;
            Some(v)
        };
        let take_f32 = |off: &mut usize| -> Option<f32> {
            let e = need(*off, 4)?;
            let v = f32::from_le_bytes(payload[*off..e].try_into().ok()?);
            *off = e;
            Some(v)
        };
        let seq = take_u64(&mut off)?;
        let surprise = take_f32(&mut off)?;
        let _kind = *payload.get(off)?;
        off += 1;
        let ctx_len = take_u32(&mut off)? as usize;
        let mut context = Vec::with_capacity(ctx_len);
        for _ in 0..ctx_len {
            context.push(take_u32(&mut off)?);
        }
        let k_len = take_u32(&mut off)? as usize;
        let mut k = Vec::with_capacity(k_len);
        for _ in 0..k_len {
            k.push(take_f32(&mut off)?);
        }
        let v_len = take_u32(&mut off)? as usize;
        let mut v = Vec::with_capacity(v_len);
        for _ in 0..v_len {
            v.push(take_f32(&mut off)?);
        }
        if off != payload.len() {
            return None;
        }
        Some(WalRecord {
            seq,
            surprise,
            context,
            k,
            v,
        })
    }
}

pub struct WalWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    pub seq: u64,
    pub records_written: u64,
    pub bytes_written: u64,
    sync_every: usize,
    since_sync: usize,
}

impl WalWriter {
    /// Open (creating) `path` in append mode. `seq` continues from
    /// the number of records already in the file.
    pub fn open(path: &Path, sync_every: usize) -> std::io::Result<Self> {
        let existing = WalReader::read_all(path)
            .map(|(r, _)| r.len() as u64)
            .unwrap_or(0);
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            seq: existing,
            records_written: 0,
            bytes_written: 0,
            sync_every: sync_every.max(1),
            since_sync: 0,
        })
    }

    pub fn append(
        &mut self,
        surprise: f32,
        context: &[u32],
        k: &[f32],
        v: &[f32],
    ) -> std::io::Result<()> {
        let rec = WalRecord {
            seq: self.seq,
            surprise,
            context: context.to_vec(),
            k: k.to_vec(),
            v: v.to_vec(),
        };
        let payload = rec.encode();
        let crc = crc32(&payload);
        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.flush()?;
        self.seq += 1;
        self.records_written += 1;
        self.bytes_written += (payload.len() + 8) as u64;
        self.since_sync += 1;
        if self.since_sync >= self.sync_every {
            self.sync()?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        self.since_sync = 0;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub struct WalReader;

impl WalReader {
    /// Read all complete records. Returns `(records, truncated)` where
    /// `truncated` is true if a torn/corrupt tail was encountered.
    pub fn read_all(path: &Path) -> std::io::Result<(Vec<WalRecord>, bool)> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), false)),
            Err(e) => return Err(e),
        };
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        let mut records = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            if off + 4 > bytes.len() {
                return Ok((records, true));
            }
            let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + len + 4 > bytes.len() {
                return Ok((records, true));
            }
            let payload = &bytes[off..off + len];
            let crc_stored =
                u32::from_le_bytes(bytes[off + len..off + len + 4].try_into().unwrap());
            off += len + 4;
            if crc32(payload) != crc_stored {
                return Ok((records, true));
            }
            match WalRecord::decode(payload) {
                Some(r) => records.push(r),
                None => return Ok((records, true)),
            }
        }
        Ok((records, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cortex-wal-{name}.log"))
    }

    #[test]
    fn wal_roundtrip() {
        let path = tmp_path("roundtrip");
        let _ = std::fs::remove_file(&path);
        let mut w = WalWriter::open(&path, 100).unwrap();
        for i in 0..3u64 {
            w.append(
                0.5 + i as f32,
                &[1, 2, 3],
                &[0.1 * i as f32, 0.2],
                &[1.0, -1.0],
            )
            .unwrap();
        }
        w.sync().unwrap();
        drop(w);
        let (recs, truncated) = WalReader::read_all(&path).unwrap();
        assert!(!truncated);
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[1].seq, 1);
        assert_eq!(recs[1].context, vec![1, 2, 3]);
        assert_eq!(recs[2].k, vec![0.2, 0.2]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wal_corrupt_tail_is_tolerated() {
        let path = tmp_path("corrupt");
        let _ = std::fs::remove_file(&path);
        let mut w = WalWriter::open(&path, 100).unwrap();
        w.append(0.9, &[], &[1.0], &[2.0]).unwrap();
        drop(w);
        // Torn tail: half a record.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0x10, 0x00, 0x00, 0x00, 0xAA, 0xBB]).unwrap();
        drop(f);
        let (recs, truncated) = WalReader::read_all(&path).unwrap();
        assert_eq!(recs.len(), 1);
        assert!(
            truncated,
            "torn tail must be reported, not silently accepted"
        );
        let _ = std::fs::remove_file(&path);
    }
}
