//! `RingSlidingAttention` — INT8 ring-buffer local sliding-window
//! attention (branch A of layer 4, `CortexMemoryBlock`).
//!
//! Each head keeps a fixed ring of `W` (default 512) KV slots, stored
//! as INT8 with one FP32 scale per slot per tensor (K and V scales
//! are separate). When the ring is full, the oldest slot is overwritten
//! in place — memory is constant no matter how long the conversation
//! runs. Attention is causal by construction in streaming mode (only
//! past tokens are in the ring) and by absolute-position masking in
//! `forward_chunk`.

use cortex_kernels::bit158::BitLinearWeights;
use cortex_kernels::{make_ternary_proj, rms_norm};

/// One head's KV ring.
struct KvRing {
    window: usize,
    head_dim: usize,
    keys: Vec<i8>,   // [W][head_dim]
    values: Vec<i8>, // [W][head_dim]
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
    positions: Vec<u64>,
    head: usize, // next slot to overwrite
    len: usize,
    next_pos: u64,
}

impl KvRing {
    fn new(window: usize, head_dim: usize) -> Self {
        Self {
            window,
            head_dim,
            keys: vec![0i8; window * head_dim],
            values: vec![0i8; window * head_dim],
            k_scales: vec![1.0; window],
            v_scales: vec![1.0; window],
            positions: vec![0; window],
            head: 0,
            len: 0,
            next_pos: 0,
        }
    }

    fn quantize(x: &[f32]) -> (Vec<i8>, f32) {
        BitLinearWeights::quantize_activation(x)
    }

    /// Append one (k, v) pair, overwriting the oldest slot when full.
    /// Returns the absolute position assigned to this token.
    fn push(&mut self, k: &[f32], v: &[f32]) -> u64 {
        let (qk, ks) = Self::quantize(k);
        let (qv, vs) = Self::quantize(v);
        let slot = self.head;
        self.keys[slot * self.head_dim..(slot + 1) * self.head_dim].copy_from_slice(&qk);
        self.values[slot * self.head_dim..(slot + 1) * self.head_dim].copy_from_slice(&qv);
        self.k_scales[slot] = ks;
        self.v_scales[slot] = vs;
        self.positions[slot] = self.next_pos;
        self.head = (self.head + 1) % self.window;
        self.len = (self.len + 1).min(self.window);
        self.next_pos += 1;
        self.positions[slot]
    }

    /// Attention output for query `q` at absolute position `pos`
    /// (only slots with `slot_pos <= pos` are attended to).
    fn attend(&self, q: &[f32], pos: u64, out: &mut [f32]) {
        out.fill(0.0);
        if self.len == 0 {
            return;
        }
        // Slots in chronological order: oldest first.
        let start = if self.len < self.window { 0 } else { self.head };
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.len);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for i in 0..self.len {
            let slot = (start + i) % self.window;
            if self.positions[slot] > pos {
                continue;
            }
            let mut dot = 0.0f32;
            for c in 0..self.head_dim {
                dot += q[c] * self.keys[slot * self.head_dim + c] as f32 * self.k_scales[slot];
            }
            scores.push((slot, dot * scale));
        }
        if scores.is_empty() {
            return;
        }
        let max_s = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        let mut weights = vec![0.0f32; scores.len()];
        for (i, (_, s)) in scores.iter().enumerate() {
            weights[i] = (s - max_s).exp();
            denom += weights[i];
        }
        for (i, (slot, _)) in scores.iter().enumerate() {
            let w = weights[i] / denom;
            let slot = *slot;
            for c in 0..self.head_dim {
                out[c] += w * self.values[slot * self.head_dim + c] as f32 * self.v_scales[slot];
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        self.keys.len() + self.values.len() + (self.k_scales.len() + self.v_scales.len()) * 4
    }
}

#[derive(Clone, Debug)]
pub struct RingAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
}

impl RingAttentionConfig {
    pub fn new(d_model: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            d_model,
            num_heads,
            head_dim,
            window_size: 512,
        }
    }
}

/// Local sliding-window attention with a fixed INT8 KV ring per head.
pub struct RingSlidingAttention {
    pub cfg: RingAttentionConfig,
    pub rms_weight: Vec<f32>,
    pub q_proj: BitLinearWeights,
    pub k_proj: BitLinearWeights,
    pub v_proj: BitLinearWeights,
    pub o_proj: BitLinearWeights,
    rings: Vec<KvRing>,
    scratch_h: Vec<f32>,
    scratch_q: Vec<f32>,
    scratch_k: Vec<f32>,
    scratch_v: Vec<f32>,
    scratch_heads: Vec<f32>,
    scratch_out: Vec<f32>,
}

impl RingSlidingAttention {
    pub fn new(
        cfg: RingAttentionConfig,
        rms_weight: Vec<f32>,
        q_proj: BitLinearWeights,
        k_proj: BitLinearWeights,
        v_proj: BitLinearWeights,
        o_proj: BitLinearWeights,
    ) -> Self {
        let rings = (0..cfg.num_heads)
            .map(|_| KvRing::new(cfg.window_size, cfg.head_dim))
            .collect();
        Self {
            scratch_h: vec![0.0; cfg.d_model],
            scratch_q: vec![0.0; cfg.num_heads * cfg.head_dim],
            scratch_k: vec![0.0; cfg.num_heads * cfg.head_dim],
            scratch_v: vec![0.0; cfg.num_heads * cfg.head_dim],
            scratch_heads: vec![0.0; cfg.num_heads * cfg.head_dim],
            scratch_out: vec![0.0; cfg.d_model],
            cfg,
            rms_weight,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rings,
        }
    }

    pub fn new_toy(cfg: RingAttentionConfig, seed: u64) -> Self {
        let d = cfg.d_model;
        let total = cfg.num_heads * cfg.head_dim;
        let scale = (1.0 / d as f32).sqrt() * 0.5;
        Self::new(
            cfg,
            vec![1.0; d],
            make_ternary_proj(total, d, seed + 1, scale),
            make_ternary_proj(total, d, seed + 2, scale),
            make_ternary_proj(total, d, seed + 3, scale),
            make_ternary_proj(d, total, seed + 4, scale),
        )
    }

    pub fn reset(&mut self) {
        for r in self.rings.iter_mut() {
            *r = KvRing::new(self.cfg.window_size, self.cfg.head_dim);
        }
    }

    /// Total KV memory in bytes — constant in sequence length.
    pub fn memory_bytes(&self) -> usize {
        self.rings.iter().map(|r| r.memory_bytes()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.rings.first().map(|r| r.len).unwrap_or(0)
    }

    pub fn window_size(&self) -> usize {
        self.cfg.window_size
    }

    /// One streaming token: `y = x + W_o @ Attn(RMSNorm(x))`.
    /// The current token's KV is written to the ring before attending,
    /// so self-attention is included and causality holds.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cfg.d_model);
        assert_eq!(out.len(), self.cfg.d_model);
        {
            let w = &self.rms_weight;
            let h = &mut self.scratch_h;
            rms_norm(w, x, h);
        }
        let h = self.scratch_h.clone();
        self.q_proj.forward_gemv(&h, &mut self.scratch_q);
        self.k_proj.forward_gemv(&h, &mut self.scratch_k);
        self.v_proj.forward_gemv(&h, &mut self.scratch_v);
        let (dk, nh) = (self.cfg.head_dim, self.cfg.num_heads);
        // Push first, then attend (pos = token index of this step).
        let mut pos = 0u64;
        for head in 0..nh {
            let k = &self.scratch_k[head * dk..(head + 1) * dk];
            let v = &self.scratch_v[head * dk..(head + 1) * dk];
            // Clone k/v out of the scratch borrows before mutating rings.
            let (kc, vc) = (k.to_vec(), v.to_vec());
            pos = self.rings[head].push(&kc, &vc);
        }
        for head in 0..nh {
            let q = self.scratch_q[head * dk..(head + 1) * dk].to_vec();
            let mut o = vec![0.0f32; dk];
            self.rings[head].attend(&q, pos, &mut o);
            self.scratch_heads[head * dk..(head + 1) * dk].copy_from_slice(&o);
        }
        let heads = self.scratch_heads.clone();
        self.o_proj.forward_gemv(&heads, &mut self.scratch_out);
        for i in 0..self.cfg.d_model {
            out[i] = x[i] + self.scratch_out[i];
        }
    }

    /// Prefill over `[L][d_model]` (sequential over tokens; the ring
    /// makes chunk parallelism unnecessary for correctness).
    pub fn forward_chunk(&mut self, xs: &[f32], outs: &mut [f32]) {
        let d = self.cfg.d_model;
        let l = xs.len() / d;
        assert_eq!(xs.len(), l * d);
        assert_eq!(outs.len(), l * d);
        for t in 0..l {
            let (x, o) = (&xs[t * d..(t + 1) * d], &mut outs[t * d..(t + 1) * d]);
            let xc = x.to_vec();
            let mut oc = vec![0.0f32; d];
            self.forward_step(&xc, &mut oc);
            o.copy_from_slice(&oc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_kernels::xorshift64star;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed ^ 0x1234_5678_9ABC_DEF0;
        (0..n)
            .map(|_| {
                let u = (xorshift64star(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn ring_overwrites_oldest_and_memory_is_constant() {
        let cfg = RingAttentionConfig {
            window_size: 4,
            ..RingAttentionConfig::new(16, 2, 8)
        };
        let mut attn = RingSlidingAttention::new_toy(cfg, 1);
        let bytes_before = attn.memory_bytes();
        let mut out = vec![0.0f32; 16];
        for t in 0..100 {
            let x = rand_vec(16, 100 + t as u64);
            attn.forward_step(&x, &mut out);
            assert!(out.iter().all(|v| v.is_finite()));
        }
        assert_eq!(attn.len(), 4, "ring must cap at window_size");
        assert_eq!(
            attn.memory_bytes(),
            bytes_before,
            "ring memory must be constant"
        );
    }

    #[test]
    fn ring_attention_recalls_matching_key() {
        // Direct ring test: push k -> v, then attend with q = k.
        // With a single entry, softmax is 1.0 and the output must be v.
        let mut ring = KvRing::new(4, 4);
        let k = vec![0.5, -1.0, 0.25, 0.75];
        let v = vec![1.0, 2.0, -1.0, 0.5];
        let pos = ring.push(&k, &v);
        let mut out = vec![0.0; 4];
        ring.attend(&k, pos, &mut out);
        for i in 0..4 {
            // INT8 quantisation of k/v introduces small error; allow 2%.
            assert!(
                (out[i] - v[i]).abs() < 0.05,
                "recall failed at {i}: {} vs {}",
                out[i],
                v[i]
            );
        }
    }

    #[test]
    fn ring_chunk_matches_step() {
        let cfg = RingAttentionConfig::new(16, 2, 8);
        let mut a = RingSlidingAttention::new_toy(cfg.clone(), 9);
        let mut b = RingSlidingAttention::new_toy(cfg, 9);
        let l = 6usize;
        let xs = rand_vec(l * 16, 3);
        let mut out_step = vec![0.0f32; l * 16];
        for t in 0..l {
            a.forward_step(
                &xs[t * 16..(t + 1) * 16],
                &mut out_step[t * 16..(t + 1) * 16],
            );
        }
        let mut out_chunk = vec![0.0f32; l * 16];
        b.forward_chunk(&xs, &mut out_chunk);
        let err = out_step
            .iter()
            .zip(out_chunk.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "chunk vs step err {err}");
    }
}
