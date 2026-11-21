//! Versioned hippocampus snapshots (Phase 3).
//!
//! Format (little-endian):
//! ```text
//! magic[4] = "HIP1", version u32 = 1, dim u32, hidden_dim u32,
//! snapshot_version u64, w1 f32[hidden*dim], w2 f32[dim*hidden],
//! crc32 u32 (IEEE, over everything after the magic)
//! ```
//! Snapshots live in a directory as `hippo-v{version}.snap`.
//! `SnapshotStore::save` assigns `max(existing) + 1`; `rollback`
//! loads a chosen version so the runtime can restore yesterday's
//! brain state with one call.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use cortex_kernels::crc32;

use crate::HippocampusWeights;

pub const SNAPSHOT_MAGIC: &[u8; 4] = b"HIP1";
pub const SNAPSHOT_VERSION: u32 = 1;

pub fn save_snapshot(
    weights: &HippocampusWeights,
    path: &Path,
    version: u64,
) -> std::io::Result<()> {
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    body.extend_from_slice(&(weights.dim as u32).to_le_bytes());
    body.extend_from_slice(&(weights.hidden_dim as u32).to_le_bytes());
    body.extend_from_slice(&version.to_le_bytes());
    for x in weights.w1.iter().chain(weights.w2.iter()) {
        body.extend_from_slice(&x.to_le_bytes());
    }
    let crc = crc32(&body);
    let mut f = File::create(path)?;
    f.write_all(SNAPSHOT_MAGIC)?;
    f.write_all(&body)?;
    f.write_all(&crc.to_le_bytes())?;
    f.sync_all()?;
    Ok(())
}

pub fn load_snapshot(path: &Path) -> std::io::Result<(HippocampusWeights, u64)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 4 + 24 + 4 || &bytes[0..4] != SNAPSHOT_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad snapshot magic",
        ));
    }
    let body = &bytes[4..bytes.len() - 4];
    let crc_stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    if crc32(body) != crc_stored {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot CRC mismatch",
        ));
    }
    let version_field = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if version_field != SNAPSHOT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported snapshot version {version_field}"),
        ));
    }
    let dim = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    let hidden_dim = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
    let version = u64::from_le_bytes(body[12..20].try_into().unwrap());
    let mut off = 20usize;
    let read_vec = |n: usize, off: &mut usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(f32::from_le_bytes(
                body[*off + i * 4..*off + i * 4 + 4].try_into().unwrap(),
            ));
        }
        *off += n * 4;
        v
    };
    let w1 = read_vec(hidden_dim * dim, &mut off);
    let w2 = read_vec(dim * hidden_dim, &mut off);
    Ok((
        HippocampusWeights {
            dim,
            hidden_dim,
            w1,
            w2,
            version: version as usize,
        },
        version,
    ))
}

pub struct SnapshotStore {
    pub dir: PathBuf,
}

impl SnapshotStore {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    fn path_for(&self, version: u64) -> PathBuf {
        self.dir.join(format!("hippo-v{version}.snap"))
    }

    pub fn list_versions(&self) -> Vec<u64> {
        let mut vs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                if let Some(name) = e.file_name().to_str() {
                    if let Some(rest) = name
                        .strip_prefix("hippo-v")
                        .and_then(|r| r.strip_suffix(".snap"))
                    {
                        if let Ok(v) = rest.parse::<u64>() {
                            vs.push(v);
                        }
                    }
                }
            }
        }
        vs.sort_unstable();
        vs
    }

    pub fn save(&self, weights: &HippocampusWeights) -> std::io::Result<u64> {
        let version = self.list_versions().last().copied().unwrap_or(0) + 1;
        save_snapshot(weights, &self.path_for(version), version)?;
        Ok(version)
    }

    pub fn load(&self, version: u64) -> std::io::Result<HippocampusWeights> {
        load_snapshot(&self.path_for(version)).map(|(w, _)| w)
    }

    pub fn latest(&self) -> std::io::Result<Option<HippocampusWeights>> {
        match self.list_versions().last() {
            Some(&v) => Ok(Some(self.load(v)?)),
            None => Ok(None),
        }
    }

    /// Roll back to `version`, returning the restored weights.
    pub fn rollback(&self, version: u64) -> std::io::Result<HippocampusWeights> {
        self.load(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_and_rollback() {
        let dir = std::env::temp_dir().join("cortex-snap-test");
        let _ = std::fs::remove_dir_all(&dir);
        let store = SnapshotStore::new(&dir).unwrap();
        let w1 = HippocampusWeights::new_initialized(8, 16, 1);
        let v1 = store.save(&w1).unwrap();
        assert_eq!(v1, 1);
        let mut w2 = w1.clone();
        for x in w2.w1.iter_mut() {
            *x += 1.0;
        }
        let v2 = store.save(&w2).unwrap();
        assert_eq!(v2, 2);
        assert_eq!(store.list_versions(), vec![1, 2]);
        let back = store.rollback(1).unwrap();
        assert_eq!(back.w1, w1.w1);
        assert_eq!(back.w2, w1.w2);
        let latest = store.latest().unwrap().unwrap();
        assert_eq!(latest.w1, w2.w1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
