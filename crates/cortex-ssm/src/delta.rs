//! `BitDeltaNetLayer` — Gated DeltaNet, the SSM layers 1–3.
//!
//! This is the spec-named layer (Phase 2). It wraps the tested
//! `cortex_kernels::GatedDeltaNetLayer` rather than duplicating its
//! chunkwise mathematics, and fixes the head geometry to square heads
//! (`head_dim_k == head_dim_v == head_dim`), so the state is exactly
//! `[num_heads, head_dim, head_dim]` as specified.
//!
//! # Transpose convention (read this before comparing with the spec)
//!
//! The spec writes the update as
//! `S_t = α S_{t-1} (I − β k kᵀ) + β v kᵀ` and the output as
//! `o_t = S_tᵀ q_t`, and also instructs the rank-1 decomposition
//! `u = S_{t-1} k_t`, `S_t = α S_{t-1} + β (v_t − α u) k_tᵀ`.
//! Those two cannot both hold for the same storage orientation:
//! `u = S k` requires `S: [dv, dk]` (and then `o = S q`), while
//! `o = Sᵀ q` requires `S: [dk, dv]` (and then `u = Sᵀ k`).
//! We implement the decomposition literally — `S` is stored
//! `[dv, dk]` (row = value dim, col = key dim) and `o = S @ q` —
//! because it is the form whose one-shot recall is verifiable:
//! after a `β = 1` write of `k -> v`, querying with `q = k` returns
//! `v` exactly. Under transposed storage that same operation is
//! `Sᵀ q`, so the two notations describe the same mathematics with
//! opposite storage. Do not "fix" the transpose without also
//! changing `u` and the rank-1 update, or recall will break.

use cortex_kernels::{GatedDeltaNetConfig, GatedDeltaNetLayer};

#[derive(Clone, Debug)]
pub struct BitDeltaNetConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub chunk_size: usize,
}

impl BitDeltaNetConfig {
    pub fn new(d_model: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            d_model,
            num_heads,
            head_dim,
            chunk_size: 64,
        }
    }
    /// Total recurrent state in bytes: heads * head_dim^2 * 4.
    pub fn state_bytes(&self) -> usize {
        self.num_heads * self.head_dim * self.head_dim * 4
    }
}

/// One Gated DeltaNet layer (spec name `BitDeltaNetLayer`).
///
/// Projections are all `BitLinearWeights` (1.58-bit): `q/k/v/o` plus
/// lightweight per-head gate projections producing `β_t, α_t`.
pub struct BitDeltaNetLayer {
    pub cfg: BitDeltaNetConfig,
    inner: GatedDeltaNetLayer,
}

impl BitDeltaNetLayer {
    pub fn new(cfg: BitDeltaNetConfig, inner: GatedDeltaNetLayer) -> Self {
        Self { cfg, inner }
    }

    /// Toy constructor with deterministic ternary projections.
    pub fn new_toy(cfg: BitDeltaNetConfig, seed: u64) -> Self {
        let inner_cfg = GatedDeltaNetConfig {
            chunk_size: cfg.chunk_size,
            ..GatedDeltaNetConfig::new(cfg.d_model, cfg.num_heads, cfg.head_dim, cfg.head_dim)
        };
        Self {
            cfg,
            inner: GatedDeltaNetLayer::new_toy(inner_cfg, seed),
        }
    }

    pub fn reset_states(&mut self) {
        self.inner.reset_states();
    }

    /// State size in bytes — constant in sequence length.
    pub fn state_bytes(&self) -> usize {
        self.inner.state_bytes()
    }

    /// Flat state in `[head, row, col]` order (`row` = value dim,
    /// `col` = key dim), i.e. the spec's `[num_heads, head_dim, head_dim]`.
    pub fn state_flat(&self) -> Vec<f32> {
        self.inner
            .states
            .iter()
            .flat_map(|s| s.s.iter().copied())
            .collect()
    }

    /// One streaming token: `y = x + W_o @ Delta(RMSNorm(x))`.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) {
        self.inner.forward_step(x, out);
    }

    /// Prefill over `[L][d_model]`; must match repeated `forward_step`.
    pub fn forward_chunk(&mut self, xs: &[f32], outs: &mut [f32]) {
        self.inner.forward_chunk(xs, outs);
    }

    /// Access to the inner layer (e.g. for `.ctxb` serialisation).
    pub fn inner(&self) -> &GatedDeltaNetLayer {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut GatedDeltaNetLayer {
        &mut self.inner
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
    fn test_deltanet_constant_memory() {
        // DoD 2: 2,048 streaming steps, state bytes constant, no NaN/Inf.
        let cfg = BitDeltaNetConfig::new(64, 4, 16);
        let mut layer = BitDeltaNetLayer::new_toy(cfg, 11);
        assert_eq!(layer.state_bytes(), 4 * 16 * 16 * 4);
        let bytes_before = layer.state_bytes();
        let mut x = rand_vec(64, 1);
        let mut out = vec![0.0f32; 64];
        for step in 0..2048 {
            // Vary the input so the state is genuinely exercised.
            for v in x.iter_mut() {
                *v = (*v * 1.0001 + 0.001 * ((step % 7) as f32)).clamp(-2.0, 2.0);
            }
            layer.forward_step(&x, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "NaN/Inf at step {step}");
            assert_eq!(
                layer.state_bytes(),
                bytes_before,
                "state grew at step {step}"
            );
        }
        assert_eq!(layer.state_flat().len(), 4 * 16 * 16);
    }

    #[test]
    fn delta_chunk_matches_step() {
        let cfg = BitDeltaNetConfig {
            chunk_size: 4,
            ..BitDeltaNetConfig::new(32, 2, 8)
        };
        let mut a = BitDeltaNetLayer::new_toy(cfg.clone(), 3);
        let mut b = BitDeltaNetLayer::new_toy(cfg, 3);
        let l = 10usize;
        let xs = rand_vec(l * 32, 5);
        let mut out_step = vec![0.0f32; l * 32];
        for t in 0..l {
            a.forward_step(
                &xs[t * 32..(t + 1) * 32],
                &mut out_step[t * 32..(t + 1) * 32],
            );
        }
        let mut out_chunk = vec![0.0f32; l * 32];
        b.forward_chunk(&xs, &mut out_chunk);
        let err = out_step
            .iter()
            .zip(out_chunk.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "chunk vs step err {err}");
    }
}
