//! Sleep-time replay: 80% WAL high-surprise samples + 20% anchors.

use std::path::Path;

use cortex_hippocampus::wal::{WalReader, WalRecord};

use crate::mezo::MezoRng;

/// A replay batch source: today's WAL records plus anti-forgetting
/// anchor records (fixed, curated samples that must not be forgotten).
pub struct ReplayBuffer {
    pub wal_records: Vec<WalRecord>,
    pub anchors: Vec<WalRecord>,
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self {
            wal_records: Vec::new(),
            anchors: Vec::new(),
        }
    }

    pub fn load_wal(&mut self, path: &Path) -> std::io::Result<usize> {
        let (recs, _truncated) = WalReader::read_all(path)?;
        let n = recs.len();
        self.wal_records.extend(recs);
        Ok(n)
    }

    pub fn add_anchor(&mut self, rec: WalRecord) {
        self.anchors.push(rec);
    }

    /// Draw a mixed batch of `batch_size` records: ~80% WAL, ~20%
    /// anchors, shuffled with `seed`. Returns the WAL fraction actually
    /// drawn (exactly 0.8 when both pools are large enough).
    pub fn mixed_batch(&self, batch_size: usize, seed: u64) -> (Vec<WalRecord>, f32) {
        if batch_size == 0 || (self.wal_records.is_empty() && self.anchors.is_empty()) {
            return (Vec::new(), 0.0);
        }
        let n_wal = ((batch_size as f32) * 0.8).round() as usize;
        let n_wal = n_wal.min(self.wal_records.len());
        let n_anchor = (batch_size - n_wal).min(self.anchors.len());
        let mut rng = MezoRng::new(seed);
        let mut pick = |pool: &[WalRecord], n: usize| -> Vec<WalRecord> {
            let mut idx: Vec<usize> = (0..pool.len()).collect();
            // Fisher–Yates with the deterministic RNG.
            for i in (1..idx.len()).rev() {
                let u = (rng.next_uniform() * (i + 1) as f32) as usize;
                idx.swap(i, u.min(i));
            }
            idx.into_iter().take(n).map(|i| pool[i].clone()).collect()
        };
        let mut batch = pick(&self.wal_records, n_wal);
        batch.extend(pick(&self.anchors, n_anchor));
        // Shuffle the combined batch.
        for i in (1..batch.len()).rev() {
            let u = (rng.next_uniform() * (i + 1) as f32) as usize;
            batch.swap(i, u.min(i));
        }
        let wal_frac = if batch.is_empty() {
            0.0
        } else {
            n_wal as f32 / batch.len() as f32
        };
        (batch, wal_frac)
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(seq: u64) -> WalRecord {
        WalRecord {
            seq,
            surprise: 0.5,
            context: vec![],
            k: vec![0.1],
            v: vec![0.2],
        }
    }

    #[test]
    fn replay_mix_is_80_20() {
        let mut buf = ReplayBuffer::new();
        for i in 0..100 {
            buf.wal_records.push(rec(i));
        }
        for i in 0..100 {
            buf.anchors.push(rec(1000 + i));
        }
        let (batch, frac) = buf.mixed_batch(50, 42);
        assert_eq!(batch.len(), 50);
        assert!(
            (frac - 0.8).abs() < 1e-6,
            "WAL fraction should be 0.8, got {frac}"
        );
    }
}
