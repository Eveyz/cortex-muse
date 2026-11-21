//! Byte-fallback BPE tokenizer (Phase 5, zero external deps).
//!
//! * Base vocabulary: the 256 UTF-8 bytes (token id = byte value), so
//!   *any* string encodes without `<UNK>` — a rare character simply
//!   falls back to its 3 UTF-8 bytes.
//! * Special tokens: `<BOS>=256`, `<EOS>=257`, `<PAD>=258`, `<UNK>=259`.
//! * Optional merge rules extend the vocabulary: each rule merges an
//!   adjacent pair `(left, right)` into a new token id. Rules are
//!   applied greedily by rank (lowest rank first), exactly like BPE.
//! * Merge file format (one rule per line, `#` = comment):
//!   ```text
//!   # left_id right_id merged_id
//!   104 101 260
//!   ```
//!   i.e. byte `h` (104) + byte `e` (101) merge into token 260.
//!   `vocab_size = 260 + merges.len()`.
//! * `decode` expands merged tokens recursively back to bytes.

use std::collections::HashMap;
use std::path::Path;

pub const BOS_ID: u32 = 256;
pub const EOS_ID: u32 = 257;
pub const PAD_ID: u32 = 258;
pub const UNK_ID: u32 = 259;
pub const NUM_SPECIAL_TOKENS: u32 = 4;
pub const BASE_VOCAB: u32 = 256;

#[derive(Clone, Debug)]
pub struct MergeRule {
    pub left: u32,
    pub right: u32,
    pub merged: u32,
    pub rank: usize,
}

#[derive(Clone, Debug)]
pub struct ByteBpeTokenizer {
    merges: Vec<MergeRule>,
    /// (left, right) -> (merged, rank)
    by_pair: HashMap<(u32, u32), (u32, usize)>,
    /// merged id -> (left, right), for decoding.
    by_merged: HashMap<u32, (u32, u32)>,
}

impl ByteBpeTokenizer {
    pub fn new() -> Self {
        Self {
            merges: Vec::new(),
            by_pair: HashMap::new(),
            by_merged: HashMap::new(),
        }
    }

    /// Add one merge rule; returns the new token id.
    pub fn add_merge(&mut self, left: u32, right: u32) -> u32 {
        let merged = BASE_VOCAB + NUM_SPECIAL_TOKENS + self.merges.len() as u32;
        let rank = self.merges.len();
        self.merges.push(MergeRule {
            left,
            right,
            merged,
            rank,
        });
        self.by_pair.insert((left, right), (merged, rank));
        self.by_merged.insert(merged, (left, right));
        merged
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn vocab_size(&self) -> usize {
        (BASE_VOCAB + NUM_SPECIAL_TOKENS + self.merges.len() as u32) as usize
    }

    /// Load merge rules from a file (see the module docs for the format).
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut tok = Self::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 3 {
                continue;
            }
            let (Ok(l), Ok(r), Ok(_m)) = (
                parts[0].parse::<u32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            ) else {
                continue;
            };
            // The merged id in the file is advisory: ids are assigned
            // sequentially in file order so the vocabulary stays dense.
            let _ = tok.add_merge(l, r);
        }
        Ok(tok)
    }

    /// Encode a string to token ids (bytes first, then BPE merges).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        if self.merges.is_empty() {
            return ids;
        }
        loop {
            // Find the lowest-rank mergeable adjacent pair.
            let mut best: Option<(usize, usize)> = None; // (position, rank)
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(_, rank)) = self.by_pair.get(&(ids[i], ids[i + 1])) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((pos, _)) = best else { break };
            let (merged, _) = self.by_pair[&(ids[pos], ids[pos + 1])];
            ids.splice(pos..pos + 2, [merged]);
        }
        ids
    }

    fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if id < BASE_VOCAB {
            out.push(id as u8);
        } else if let Some(&(l, r)) = self.by_merged.get(&id) {
            self.token_bytes(l, out);
            self.token_bytes(r, out);
        } else {
            // Special token or unknown id: emit nothing (they carry no bytes).
        }
    }

    /// Decode token ids back to a string (lossy for invalid UTF-8).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.token_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Default for ByteBpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_fallback_roundtrip() {
        let tok = ByteBpeTokenizer::new();
        assert_eq!(tok.vocab_size(), 260);
        let text = "Hello, 世界! 🦀 Rust";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len());
        assert_eq!(tok.decode(&ids), text);
        // A rare character never produces UNK: it is just bytes.
        let rare = "𐍈";
        let ids = tok.encode(rare);
        assert_eq!(ids.len(), 4);
        assert_eq!(tok.decode(&ids), rare);
    }

    #[test]
    fn merge_rules_apply_by_rank() {
        let mut tok = ByteBpeTokenizer::new();
        let he = tok.add_merge(b'h' as u32, b'e' as u32);
        let _hel = tok.add_merge(he, b'l' as u32);
        let ids = tok.encode("hello");
        // "hel" + "l" + "o"
        assert_eq!(ids.len(), 3);
        assert_eq!(tok.decode(&ids), "hello");
        assert_eq!(tok.encode("he"), vec![he]);
    }

    #[test]
    fn merges_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("cortex-merges-test.txt");
        std::fs::write(&path, "# comment\n104 101 260\n").unwrap();
        let tok = ByteBpeTokenizer::from_file(&path).unwrap();
        assert_eq!(tok.num_merges(), 1);
        assert_eq!(tok.encode("he"), vec![260]);
        let _ = std::fs::remove_file(&path);
    }
}
