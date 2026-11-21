//! `CortexModel` — full assembly (Phase 5).
//!
//! One macro-block = 3x `BitSSMBlock` (Gated DeltaNet + SwiGLU FFN)
//!   + 1x `CortexMemoryBlock` (ring attention branch A + hippocampus
//!     branch B, sequentially composed, + SwiGLU FFN). `N_MACROS` macros
//!     are stacked; the embedding and the LM head are weight-tied
//!     (one INT8 ternary matrix + per-row FP32 scales).
//!
//! Configurations:
//! * `CortexConfig::micro()` — `N_MACROS=2` (8 layers), `d_model=256`,
//!   used by every test and by the CLI default.
//! * `CortexConfig::production()` — `N_MACROS=6` (24 layers),
//!   `d_model=2048`, `vocab_size=32768`. This is a *configuration*;
//!   instantiating it allocates the full toy weight set and is not
//!   done by any test.
//!
//! `φ` (for MeZO) is collected in one fixed order, documented on
//! [`CortexModel::collect_phi`]; `apply_phi` walks the same order.
//! `loss_for_sequences` disables hippocampus learning while it runs,
//! so the loss is a pure function of `φ`.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use cortex_hippocampus::{CortexMemoryLayer, HippocampusWeights, PlasticityConfig, SnapshotStore};
use cortex_kernels::bit158::BitLinearWeights;
use cortex_kernels::ctxb::{read_ctxb, write_ctxb, CtxbHeader, CtxbTensor, CtxbTensorKind};
use cortex_kernels::{make_ternary_proj, rms_norm, BitSwiGLU};
use cortex_sleep::consolidate_hippocampus;
use cortex_ssm::{BitDeltaNetConfig, BitDeltaNetLayer, RingAttentionConfig, RingSlidingAttention};

use crate::tokenizer::{EOS_ID, UNK_ID};

#[derive(Clone, Debug)]
pub struct CortexConfig {
    pub d_model: usize,
    pub num_macros: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hippo_hidden_dim: usize,
    pub ffn_hidden_dim: usize,
    pub window_size: usize,
    pub chunk_size: usize,
}

impl CortexConfig {
    /// Micro config: 2 macros = 8 layers, d_model 256, vocab 260
    /// (256 bytes + 4 specials, no merges). Used by tests and the CLI.
    pub fn micro() -> Self {
        Self {
            d_model: 256,
            num_macros: 2,
            num_heads: 4,
            head_dim: 64,
            vocab_size: 260,
            hippo_hidden_dim: 512,
            ffn_hidden_dim: 512,
            window_size: 512,
            chunk_size: 64,
        }
    }

    /// Production config: 6 macros = 24 layers, d_model 2048.
    pub fn production() -> Self {
        Self {
            d_model: 2048,
            num_macros: 6,
            num_heads: 32,
            head_dim: 64,
            vocab_size: 32768,
            hippo_hidden_dim: 4096,
            ffn_hidden_dim: 5504,
            window_size: 512,
            chunk_size: 64,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_macros * 4
    }
}

fn ternary(rows: usize, cols: usize, seed: u64) -> BitLinearWeights {
    let scale = (1.0 / cols as f32).sqrt() * 0.5;
    make_ternary_proj(rows, cols, seed, scale)
}

/// Layers 1–3 of a macro: Gated DeltaNet + SwiGLU FFN (pre-norm).
pub struct BitSSMBlock {
    pub delta: BitDeltaNetLayer,
    pub ffn: BitSwiGLU,
    pub rms_ffn: Vec<f32>,
}

impl BitSSMBlock {
    pub fn new_toy(cfg: &CortexConfig, seed: u64) -> Self {
        let delta_cfg = BitDeltaNetConfig {
            chunk_size: cfg.chunk_size,
            ..BitDeltaNetConfig::new(cfg.d_model, cfg.num_heads, cfg.head_dim)
        };
        let delta = BitDeltaNetLayer::new_toy(delta_cfg, seed);
        let ffn = BitSwiGLU::new(
            ternary(cfg.ffn_hidden_dim, cfg.d_model, seed + 10),
            ternary(cfg.ffn_hidden_dim, cfg.d_model, seed + 11),
            ternary(cfg.d_model, cfg.ffn_hidden_dim, seed + 12),
        );
        Self {
            delta,
            ffn,
            rms_ffn: vec![1.0; cfg.d_model],
        }
    }

    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) {
        let d = x.len();
        let mut y = vec![0.0f32; d];
        self.delta.forward_step(x, &mut y);
        let mut h = vec![0.0f32; d];
        rms_norm(&self.rms_ffn, &y, &mut h);
        let mut f = vec![0.0f32; d];
        self.ffn.forward(&h, &mut f);
        for i in 0..d {
            out[i] = y[i] + f[i];
        }
    }

    pub fn reset_states(&mut self) {
        self.delta.reset_states();
    }
}

/// Layer 4 of a macro: ring attention (A) + hippocampus (B) + FFN.
pub struct CortexMemoryBlock {
    pub ring: RingSlidingAttention,
    pub hippo: CortexMemoryLayer,
    pub ffn: BitSwiGLU,
    pub rms_ffn: Vec<f32>,
}

impl CortexMemoryBlock {
    pub fn new_toy(cfg: &CortexConfig, seed: u64) -> Self {
        let d = cfg.d_model;
        let ring_cfg = RingAttentionConfig {
            window_size: cfg.window_size,
            ..RingAttentionConfig::new(d, cfg.num_heads, cfg.head_dim)
        };
        let ring = RingSlidingAttention::new_toy(ring_cfg, seed);
        let hippo = CortexMemoryLayer {
            dim: d,
            rms_weight: vec![1.0; d],
            q_proj: ternary(d, d, seed + 20),
            k_proj: ternary(d, d, seed + 21),
            v_proj: ternary(d, d, seed + 22),
            gate_proj: ternary(d, d, seed + 23),
            hippocampus: cortex_hippocampus::AsyncHippocampusHandle::spawn(
                HippocampusWeights::new_initialized(d, cfg.hippo_hidden_dim, seed),
                PlasticityConfig::default(),
                64,
            ),
        };
        let ffn = BitSwiGLU::new(
            ternary(cfg.ffn_hidden_dim, d, seed + 30),
            ternary(cfg.ffn_hidden_dim, d, seed + 31),
            ternary(d, cfg.ffn_hidden_dim, seed + 32),
        );
        Self {
            ring,
            hippo,
            ffn,
            rms_ffn: vec![1.0; d],
        }
    }

    pub fn forward_step(
        &mut self,
        x: &[f32],
        is_eos: bool,
        out: &mut [f32],
        learn: bool,
        hippo_enabled: bool,
    ) {
        let d = x.len();
        let mut y1 = vec![0.0f32; d];
        self.ring.forward_step(x, &mut y1);
        let mut y2 = vec![0.0f32; d];
        if !hippo_enabled {
            // Seed-kernel check mode: the hippocampus is online memory
            // and is not part of the exported .ctxb, so it is bypassed
            // entirely (y2 = y1) for a deterministic comparison.
            y2.copy_from_slice(&y1);
        } else if learn {
            self.hippo.step_token(&y1, is_eos, &mut y2);
        } else {
            self.hippo.step_token_readonly(&y1, &mut y2);
        }
        let mut h = vec![0.0f32; d];
        rms_norm(&self.rms_ffn, &y2, &mut h);
        let mut f = vec![0.0f32; d];
        self.ffn.forward(&h, &mut f);
        for i in 0..d {
            out[i] = y2[i] + f[i];
        }
    }

    pub fn reset_states(&mut self) {
        self.ring.reset();
    }
}

pub struct MacroBlock {
    pub ssm_blocks: Vec<BitSSMBlock>,
    pub memory_block: CortexMemoryBlock,
}

impl MacroBlock {
    pub fn new_toy(cfg: &CortexConfig, seed: u64) -> Self {
        let ssm_blocks = (0..3)
            .map(|i| BitSSMBlock::new_toy(cfg, seed + 100 * i as u64))
            .collect();
        let memory_block = CortexMemoryBlock::new_toy(cfg, seed + 1000);
        Self {
            ssm_blocks,
            memory_block,
        }
    }

    pub fn forward_step(
        &mut self,
        x: &[f32],
        is_eos: bool,
        out: &mut [f32],
        learn: bool,
        hippo_enabled: bool,
    ) {
        let d = x.len();
        let mut cur = x.to_vec();
        let mut nxt = vec![0.0f32; d];
        for b in self.ssm_blocks.iter_mut() {
            b.forward_step(&cur, &mut nxt);
            std::mem::swap(&mut cur, &mut nxt);
        }
        self.memory_block
            .forward_step(&cur, is_eos, &mut nxt, learn, hippo_enabled);
        out.copy_from_slice(&nxt);
    }

    pub fn reset_states(&mut self) {
        for b in self.ssm_blocks.iter_mut() {
            b.reset_states();
        }
        self.memory_block.reset_states();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

/// Sample one token id from `logits` (deterministic given `seed`).
/// Applies repetition penalty, temperature, then top-p nucleus sampling.
pub fn sample_token(
    logits: &[f32],
    params: &SamplingParams,
    recent: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let mut l = logits.to_vec();
    if params.repetition_penalty != 1.0 {
        for &t in recent {
            let t = t as usize;
            if t < l.len() {
                if l[t] > 0.0 {
                    l[t] /= params.repetition_penalty;
                } else {
                    l[t] *= params.repetition_penalty;
                }
            }
        }
    }
    if params.temperature <= 0.0 {
        return l
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    for v in l.iter_mut() {
        *v /= params.temperature;
    }
    let max_l = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = l.iter().map(|v| (v - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // Top-p: sort indices by probability descending, keep cumulative <= top_p.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut cum = 0.0f32;
    let mut keep: Vec<usize> = Vec::new();
    for &i in &order {
        keep.push(i);
        cum += probs[i];
        if cum >= params.top_p {
            break;
        }
    }
    let keep_sum: f32 = keep.iter().map(|&i| probs[i]).sum();
    // Deterministic xorshift draw.
    let mut x = *rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    let u = ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64) as f32;
    let mut acc = 0.0f32;
    for &i in &keep {
        acc += probs[i] / keep_sum;
        if u <= acc {
            return i as u32;
        }
    }
    *keep.last().unwrap() as u32
}

pub struct GenStats {
    pub tokens: usize,
    pub total_ms: f64,
    pub ms_per_token: f64,
}

pub struct CortexModel {
    pub cfg: CortexConfig,
    /// Tied embedding / LM head: [vocab_size, d_model] ternary + row scales.
    pub embedding: BitLinearWeights,
    /// Dequantised copy of `embedding`, `[vocab_size, d_model]`.
    /// The LM head dots against this on every token (66k lookups in
    /// the micro config), so going through `BitLinearWeights::get`
    /// (tile div/mod per element) made generation ~10x slower than
    /// it needs to be. Rebuilt whenever the embedding scales change.
    embedding_f32: Vec<f32>,
    pub macros: Vec<MacroBlock>,
    pub final_rms: Vec<f32>,
    pub hippo_learning: bool,
    /// If false, the hippocampus branch is bypassed (y2 = y1).
    /// Used by the `.ctxb` self-check, where the seed kernel is
    /// verified without its online memory.
    pub hippo_enabled: bool,
}

impl CortexModel {
    pub fn new_toy(cfg: CortexConfig, seed: u64) -> Self {
        let embedding = ternary(cfg.vocab_size, cfg.d_model, seed + 9000);
        let macros = (0..cfg.num_macros)
            .map(|i| MacroBlock::new_toy(&cfg, seed + 10000 * i as u64))
            .collect();
        let mut m = Self {
            cfg,
            embedding,
            embedding_f32: Vec::new(),
            macros,
            final_rms: vec![1.0; 256],
            hippo_learning: true,
            hippo_enabled: true,
        }
        .with_final_rms();
        m.rebuild_embedding_cache();
        m
    }

    /// Rebuild the dequantised embedding cache (call after any
    /// change to `embedding.scales` or `embedding` itself).
    pub fn rebuild_embedding_cache(&mut self) {
        let (vocab, d) = (self.cfg.vocab_size, self.cfg.d_model);
        self.embedding_f32 = vec![0.0f32; vocab * d];
        for v in 0..vocab {
            let scale = self.embedding.scales[v];
            for c in 0..d {
                self.embedding_f32[v * d + c] = self.embedding.get(v, c) as f32 * scale;
            }
        }
    }

    fn with_final_rms(mut self) -> Self {
        self.final_rms = vec![1.0; self.cfg.d_model];
        self
    }

    pub fn reset_states(&mut self) {
        for m in self.macros.iter_mut() {
            m.reset_states();
        }
    }

    /// Dequantised embedding row for `token` (weight-tied with LM head).
    pub fn embed_row(&self, token: u32, out: &mut [f32]) {
        let t = if (token as usize) < self.cfg.vocab_size {
            token as usize
        } else {
            UNK_ID as usize
        };
        let d = self.cfg.d_model;
        out.copy_from_slice(&self.embedding_f32[t * d..(t + 1) * d]);
    }

    /// One token forward: returns logits for the *next* token.
    pub fn forward_token(&mut self, token: u32, is_eos: bool) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut x = vec![0.0f32; d];
        self.embed_row(token, &mut x);
        let mut y = vec![0.0f32; d];
        let learn = self.hippo_learning;
        let hippo_enabled = self.hippo_enabled;
        for m in self.macros.iter_mut() {
            m.forward_step(&x, is_eos, &mut y, learn, hippo_enabled);
            std::mem::swap(&mut x, &mut y);
        }
        let mut h = vec![0.0f32; d];
        rms_norm(&self.final_rms, &x, &mut h);
        // LM head (tied): logits[v] = h . embed_row(v), over the
        // dequantised cache (contiguous f32, no tile div/mod).
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        for v in 0..self.cfg.vocab_size {
            let row = &self.embedding_f32[v * d..(v + 1) * d];
            let mut acc = 0.0f32;
            for c in 0..d {
                acc += h[c] * row[c];
            }
            logits[v] = acc;
        }
        logits
    }

    /// Cross-entropy loss over `seqs` (each seq predicts its next token).
    /// Hippocampus learning is disabled during the call so the loss is
    /// a pure function of `φ`; states are reset first.
    pub fn loss_for_sequences(&mut self, seqs: &[Vec<u32>]) -> f32 {
        let prev = self.hippo_learning;
        self.hippo_learning = false;
        self.reset_states();
        let mut total = 0.0f32;
        let mut count = 0usize;
        for seq in seqs {
            if seq.len() < 2 {
                continue;
            }
            for t in 0..seq.len() - 1 {
                let logits = self.forward_token(seq[t], t == seq.len() - 2);
                let target = seq[t + 1] as usize;
                if target >= logits.len() {
                    continue;
                }
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut denom = 0.0f32;
                for v in &logits {
                    denom += (v - max_l).exp();
                }
                let logp = (logits[target] - max_l) - denom.ln();
                total -= logp;
                count += 1;
            }
        }
        self.hippo_learning = prev;
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    /// Generate `max_new` tokens after `prompt` (prompt is prefilled
    /// through the model first). Returns `(tokens, stats)`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
    ) -> (Vec<u32>, GenStats) {
        let start = Instant::now();
        let mut rng_state = params.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out_tokens: Vec<u32> = Vec::new();
        let mut recent: Vec<u32> = prompt.to_vec();
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        if prompt.is_empty() {
            logits = self.forward_token(EOS_ID, false);
        } else {
            for (i, &t) in prompt.iter().enumerate() {
                logits = self.forward_token(t, i == prompt.len() - 1);
            }
        }
        for _ in 0..max_new {
            let next = sample_token(&logits, params, &recent, &mut rng_state);
            out_tokens.push(next);
            recent.push(next);
            if next == EOS_ID {
                break;
            }
            logits = self.forward_token(next, false);
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let n = out_tokens.len();
        let stats = GenStats {
            tokens: n,
            total_ms,
            ms_per_token: total_ms / n.max(1) as f64,
        };
        (out_tokens, stats)
    }

    /// Total recurrent state in bytes (DeltaNet states + KV rings).
    pub fn state_bytes(&self) -> usize {
        let mut n = 0usize;
        for m in &self.macros {
            for b in &m.ssm_blocks {
                n += b.delta.state_bytes();
            }
            n += m.memory_block.ring.memory_bytes();
        }
        n
    }

    pub fn hippo_version(&self) -> usize {
        // Version of the first macro's hippocampus (all macros share
        // the same versioning scheme; the CLI reports macro 0).
        self.macros[0]
            .memory_block
            .hippo
            .hippocampus
            .snapshot()
            .version
    }

    /// Sleep consolidation for macro 0's hippocampus: snapshot,
    /// orthogonalise, hot-swap, and save as a new snapshot version.
    /// Returns `(deduped_w1, deduped_w2, new_version)`.
    pub fn consolidate_hippocampus(
        &mut self,
        store: &SnapshotStore,
    ) -> std::io::Result<(usize, usize, u64)> {
        let handle = &self.macros[0].memory_block.hippo.hippocampus;
        let mut weights = handle.snapshot();
        let (d1, d2) = consolidate_hippocampus(&mut weights, 0.95);
        handle.replace_weights(weights.clone());
        let version = store.save(&weights)?;
        Ok((d1, d2, version))
    }

    /// Roll the macro-0 hippocampus back to a snapshot version.
    pub fn reset_memory_to_snapshot(
        &mut self,
        store: &SnapshotStore,
        version: u64,
    ) -> std::io::Result<()> {
        let weights = store.rollback(version)?;
        self.macros[0]
            .memory_block
            .hippo
            .hippocampus
            .replace_weights(weights);
        Ok(())
    }

    // -- φ (MeZO continuous parameters) ---------------------------------
    //
    // Fixed order (collect_phi and apply_phi must match exactly):
    // embedding scales; per macro: per SSM block (delta q/k/v/beta/
    // alpha/out scales, ffn gate/up/down scales, rms_ffn), then the
    // memory block (ring q/k/v/o scales + ring rms, hippo q/k/v/gate
    // scales + hippo rms, ffn gate/up/down scales, rms_ffn);
    // finally final_rms.

    pub fn collect_phi(&self) -> Vec<f32> {
        let mut phi = Vec::new();
        phi.extend_from_slice(&self.embedding.scales);
        for m in &self.macros {
            for b in &m.ssm_blocks {
                let inner = b.delta.inner();
                phi.extend_from_slice(&inner.q_proj.scales);
                phi.extend_from_slice(&inner.k_proj.scales);
                phi.extend_from_slice(&inner.v_proj.scales);
                phi.extend_from_slice(&inner.beta_proj.scales);
                phi.extend_from_slice(&inner.alpha_proj.scales);
                phi.extend_from_slice(&inner.out_proj.scales);
                phi.extend_from_slice(&b.ffn.gate_proj.scales);
                phi.extend_from_slice(&b.ffn.up_proj.scales);
                phi.extend_from_slice(&b.ffn.down_proj.scales);
                phi.extend_from_slice(&b.rms_ffn);
            }
            let mem = &m.memory_block;
            phi.extend_from_slice(&mem.ring.q_proj.scales);
            phi.extend_from_slice(&mem.ring.k_proj.scales);
            phi.extend_from_slice(&mem.ring.v_proj.scales);
            phi.extend_from_slice(&mem.ring.o_proj.scales);
            phi.extend_from_slice(&mem.ring.rms_weight);
            phi.extend_from_slice(&mem.hippo.q_proj.scales);
            phi.extend_from_slice(&mem.hippo.k_proj.scales);
            phi.extend_from_slice(&mem.hippo.v_proj.scales);
            phi.extend_from_slice(&mem.hippo.gate_proj.scales);
            phi.extend_from_slice(&mem.hippo.rms_weight);
            phi.extend_from_slice(&mem.ffn.gate_proj.scales);
            phi.extend_from_slice(&mem.ffn.up_proj.scales);
            phi.extend_from_slice(&mem.ffn.down_proj.scales);
            phi.extend_from_slice(&mem.rms_ffn);
        }
        phi.extend_from_slice(&self.final_rms);
        phi
    }

    pub fn apply_phi(&mut self, phi: &[f32]) {
        let mut off = 0usize;
        let take = |n: usize, off: &mut usize| -> &[f32] {
            let s = &phi[*off..*off + n];
            *off += n;
            s
        };
        let n = self.embedding.scales.len();
        self.embedding.scales.copy_from_slice(take(n, &mut off));
        for m in self.macros.iter_mut() {
            for b in m.ssm_blocks.iter_mut() {
                let inner = b.delta.inner_mut();
                let n = inner.q_proj.scales.len();
                inner.q_proj.scales.copy_from_slice(take(n, &mut off));
                let n = inner.k_proj.scales.len();
                inner.k_proj.scales.copy_from_slice(take(n, &mut off));
                let n = inner.v_proj.scales.len();
                inner.v_proj.scales.copy_from_slice(take(n, &mut off));
                let n = inner.beta_proj.scales.len();
                inner.beta_proj.scales.copy_from_slice(take(n, &mut off));
                let n = inner.alpha_proj.scales.len();
                inner.alpha_proj.scales.copy_from_slice(take(n, &mut off));
                let n = inner.out_proj.scales.len();
                inner.out_proj.scales.copy_from_slice(take(n, &mut off));
                let n = b.ffn.gate_proj.scales.len();
                b.ffn.gate_proj.scales.copy_from_slice(take(n, &mut off));
                let n = b.ffn.up_proj.scales.len();
                b.ffn.up_proj.scales.copy_from_slice(take(n, &mut off));
                let n = b.ffn.down_proj.scales.len();
                b.ffn.down_proj.scales.copy_from_slice(take(n, &mut off));
                let n = b.rms_ffn.len();
                b.rms_ffn.copy_from_slice(take(n, &mut off));
            }
            let mem = &mut m.memory_block;
            let n = mem.ring.q_proj.scales.len();
            mem.ring.q_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ring.k_proj.scales.len();
            mem.ring.k_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ring.v_proj.scales.len();
            mem.ring.v_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ring.o_proj.scales.len();
            mem.ring.o_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ring.rms_weight.len();
            mem.ring.rms_weight.copy_from_slice(take(n, &mut off));
            let n = mem.hippo.q_proj.scales.len();
            mem.hippo.q_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.hippo.k_proj.scales.len();
            mem.hippo.k_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.hippo.v_proj.scales.len();
            mem.hippo.v_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.hippo.gate_proj.scales.len();
            mem.hippo
                .gate_proj
                .scales
                .copy_from_slice(take(n, &mut off));
            let n = mem.hippo.rms_weight.len();
            mem.hippo.rms_weight.copy_from_slice(take(n, &mut off));
            let n = mem.ffn.gate_proj.scales.len();
            mem.ffn.gate_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ffn.up_proj.scales.len();
            mem.ffn.up_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.ffn.down_proj.scales.len();
            mem.ffn.down_proj.scales.copy_from_slice(take(n, &mut off));
            let n = mem.rms_ffn.len();
            mem.rms_ffn.copy_from_slice(take(n, &mut off));
        }
        let n = self.final_rms.len();
        self.final_rms.copy_from_slice(take(n, &mut off));
        assert_eq!(off, phi.len(), "phi length mismatch in apply_phi");
        self.rebuild_embedding_cache();
    }

    /// The output-scale subspace of `φ`: the tied embedding/LM-head
    /// row scales plus the final RMS weights. These are the most
    /// cross-entropy-sensitive continuous parameters (the LM head is
    /// the last linear map before the softmax), so sleep-time MeZO
    /// tunes them first. Full-`φ` MeZO is proven by the sleep crate's
    /// quadratic test; on the random toy model a single random
    /// direction in the full 24k-dim `φ` captures only ~`1/√dim` of
    /// the gradient (measured: 30 steps move the loss by ~3% at a
    /// safe lr, and diverge at lr ≥ 0.004), while this 516-dim
    /// subspace with 4 averaged directions reaches ~0.79x in 30 steps.
    pub fn collect_phi_lm(&self) -> Vec<f32> {
        let mut phi = self.embedding.scales.clone();
        phi.extend_from_slice(&self.final_rms);
        phi
    }

    pub fn apply_phi_lm(&mut self, phi: &[f32]) {
        assert_eq!(
            phi.len(),
            self.embedding.scales.len() + self.final_rms.len()
        );
        let n = self.embedding.scales.len();
        self.embedding.scales.copy_from_slice(&phi[..n]);
        self.final_rms.copy_from_slice(&phi[n..]);
        self.rebuild_embedding_cache();
    }

    // -- .ctxb serialisation -------------------------------------------
    //
    // Tensor names follow the φ order above, prefixed per macro/block.

    pub fn to_ctxb_tensors(&self) -> Vec<CtxbTensor> {
        let mut ts = Vec::new();
        ts.push(CtxbTensor::bitlinear("embedding", self.embedding.clone()));
        for (mi, m) in self.macros.iter().enumerate() {
            for (si, b) in m.ssm_blocks.iter().enumerate() {
                let inner = b.delta.inner();
                let p = format!("macro{mi}.ssm{si}.delta");
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.q_proj"),
                    inner.q_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.k_proj"),
                    inner.k_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.v_proj"),
                    inner.v_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.beta_proj"),
                    inner.beta_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.alpha_proj"),
                    inner.alpha_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.out_proj"),
                    inner.out_proj.clone(),
                ));
                let p = format!("macro{mi}.ssm{si}.ffn");
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.gate_proj"),
                    b.ffn.gate_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.up_proj"),
                    b.ffn.up_proj.clone(),
                ));
                ts.push(CtxbTensor::bitlinear(
                    format!("{p}.down_proj"),
                    b.ffn.down_proj.clone(),
                ));
                ts.push(CtxbTensor::vector(
                    format!("macro{mi}.ssm{si}.rms_ffn"),
                    b.rms_ffn.clone(),
                ));
                ts.push(CtxbTensor::vector(
                    format!("macro{mi}.ssm{si}.delta.rms"),
                    inner.rms_weight.clone(),
                ));
                ts.push(CtxbTensor::vector(
                    format!("macro{mi}.ssm{si}.delta.log_decay"),
                    inner.log_decay.clone(),
                ));
            }
            let mem = &m.memory_block;
            let p = format!("macro{mi}.mem.ring");
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.q_proj"),
                mem.ring.q_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.k_proj"),
                mem.ring.k_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.v_proj"),
                mem.ring.v_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.o_proj"),
                mem.ring.o_proj.clone(),
            ));
            ts.push(CtxbTensor::vector(
                format!("{p}.rms"),
                mem.ring.rms_weight.clone(),
            ));
            let p = format!("macro{mi}.mem.hippo");
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.q_proj"),
                mem.hippo.q_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.k_proj"),
                mem.hippo.k_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.v_proj"),
                mem.hippo.v_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.gate_proj"),
                mem.hippo.gate_proj.clone(),
            ));
            ts.push(CtxbTensor::vector(
                format!("{p}.rms"),
                mem.hippo.rms_weight.clone(),
            ));
            let p = format!("macro{mi}.mem.ffn");
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.gate_proj"),
                mem.ffn.gate_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.up_proj"),
                mem.ffn.up_proj.clone(),
            ));
            ts.push(CtxbTensor::bitlinear(
                format!("{p}.down_proj"),
                mem.ffn.down_proj.clone(),
            ));
            ts.push(CtxbTensor::vector(
                format!("macro{mi}.mem.rms_ffn"),
                mem.rms_ffn.clone(),
            ));
        }
        ts.push(CtxbTensor::vector("final_rms", self.final_rms.clone()));
        ts
    }

    pub fn save_ctxb(&self, path: &Path) -> Result<u64, cortex_kernels::CtxbError> {
        let mut header = CtxbHeader::new(
            self.cfg.d_model as u32,
            self.cfg.num_macros as u32,
            self.cfg.num_heads as u32,
            self.cfg.head_dim as u32,
            self.cfg.vocab_size as u32,
        );
        header.ffn_hidden = self.cfg.ffn_hidden_dim as u32;
        header.hippo_hidden = self.cfg.hippo_hidden_dim as u32;
        write_ctxb(path, &header, &self.to_ctxb_tensors())
    }

    /// Load a `.ctxb` file into a model. The config is taken from the
    /// file header (toy weights are replaced wholesale by the file's
    /// tensors, matched by name).
    pub fn load_ctxb(path: &Path) -> Result<Self, cortex_kernels::CtxbError> {
        let (header, tensors) = read_ctxb(path)?;
        let cfg = CortexConfig {
            d_model: header.d_model as usize,
            num_macros: header.num_macros as usize,
            num_heads: header.num_heads as usize,
            head_dim: header.head_dim as usize,
            vocab_size: header.vocab_size as usize,
            hippo_hidden_dim: header.hippo_hidden_or_default() as usize,
            ffn_hidden_dim: header.ffn_hidden_or_default() as usize,
            window_size: 512,
            chunk_size: 64,
        };
        let mut model = CortexModel::new_toy(cfg, 0);
        let mut map: HashMap<String, CtxbTensor> =
            tensors.into_iter().map(|t| (t.name.clone(), t)).collect();
        let take_bitlinear =
            |map: &mut HashMap<String, CtxbTensor>, name: &str| -> BitLinearWeights {
                match map.remove(name) {
                    Some(CtxbTensor {
                        kind: CtxbTensorKind::BitLinear(w),
                        ..
                    }) => w,
                    _ => panic!("ctxb missing BitLinear tensor {name}"),
                }
            };
        let take_vector = |map: &mut HashMap<String, CtxbTensor>, name: &str| -> Vec<f32> {
            match map.remove(name) {
                Some(CtxbTensor {
                    kind: CtxbTensorKind::F32Vector(v),
                    ..
                }) => v,
                _ => panic!("ctxb missing vector tensor {name}"),
            }
        };
        model.embedding = take_bitlinear(&mut map, "embedding");
        for (mi, m) in model.macros.iter_mut().enumerate() {
            for (si, b) in m.ssm_blocks.iter_mut().enumerate() {
                let p = format!("macro{mi}.ssm{si}.delta");
                let inner = b.delta.inner_mut();
                inner.q_proj = take_bitlinear(&mut map, &format!("{p}.q_proj"));
                inner.k_proj = take_bitlinear(&mut map, &format!("{p}.k_proj"));
                inner.v_proj = take_bitlinear(&mut map, &format!("{p}.v_proj"));
                inner.beta_proj = take_bitlinear(&mut map, &format!("{p}.beta_proj"));
                inner.alpha_proj = take_bitlinear(&mut map, &format!("{p}.alpha_proj"));
                inner.out_proj = take_bitlinear(&mut map, &format!("{p}.out_proj"));
                inner.rms_weight = take_vector(&mut map, &format!("{p}.rms"));
                inner.log_decay = take_vector(&mut map, &format!("{p}.log_decay"));
                let p = format!("macro{mi}.ssm{si}.ffn");
                b.ffn.gate_proj = take_bitlinear(&mut map, &format!("{p}.gate_proj"));
                b.ffn.up_proj = take_bitlinear(&mut map, &format!("{p}.up_proj"));
                b.ffn.down_proj = take_bitlinear(&mut map, &format!("{p}.down_proj"));
                b.rms_ffn = take_vector(&mut map, &format!("macro{mi}.ssm{si}.rms_ffn"));
            }
            let mem = &mut m.memory_block;
            let p = format!("macro{mi}.mem.ring");
            mem.ring.q_proj = take_bitlinear(&mut map, &format!("{p}.q_proj"));
            mem.ring.k_proj = take_bitlinear(&mut map, &format!("{p}.k_proj"));
            mem.ring.v_proj = take_bitlinear(&mut map, &format!("{p}.v_proj"));
            mem.ring.o_proj = take_bitlinear(&mut map, &format!("{p}.o_proj"));
            mem.ring.rms_weight = take_vector(&mut map, &format!("{p}.rms"));
            let p = format!("macro{mi}.mem.hippo");
            mem.hippo.q_proj = take_bitlinear(&mut map, &format!("{p}.q_proj"));
            mem.hippo.k_proj = take_bitlinear(&mut map, &format!("{p}.k_proj"));
            mem.hippo.v_proj = take_bitlinear(&mut map, &format!("{p}.v_proj"));
            mem.hippo.gate_proj = take_bitlinear(&mut map, &format!("{p}.gate_proj"));
            mem.hippo.rms_weight = take_vector(&mut map, &format!("{p}.rms"));
            let p = format!("macro{mi}.mem.ffn");
            mem.ffn.gate_proj = take_bitlinear(&mut map, &format!("{p}.gate_proj"));
            mem.ffn.up_proj = take_bitlinear(&mut map, &format!("{p}.up_proj"));
            mem.ffn.down_proj = take_bitlinear(&mut map, &format!("{p}.down_proj"));
            mem.rms_ffn = take_vector(&mut map, &format!("macro{mi}.mem.rms_ffn"));
        }
        model.final_rms = take_vector(&mut map, "final_rms");
        model.rebuild_embedding_cache();
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteBpeTokenizer;

    fn micro_model() -> CortexModel {
        CortexModel::new_toy(CortexConfig::micro(), 42)
    }

    #[test]
    fn model_forward_is_finite() {
        let mut m = micro_model();
        let logits = m.forward_token(65, false);
        assert_eq!(logits.len(), 260);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn model_phi_roundtrip() {
        let mut m = micro_model();
        let phi = m.collect_phi();
        assert!(!phi.is_empty());
        let mut phi2 = phi.clone();
        for v in phi2.iter_mut() {
            *v += 0.01;
        }
        m.apply_phi(&phi2);
        let back = m.collect_phi();
        for (a, b) in back.iter().zip(phi2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn model_ctxb_roundtrip_logits_match() {
        let mut m = micro_model();
        let path = std::env::temp_dir().join("cortex-model-test.ctxb");
        m.save_ctxb(&path).unwrap();
        let mut m2 = CortexModel::load_ctxb(&path).unwrap();
        // The .ctxb file holds the frozen seed kernel only: the
        // online hippocampus weights are not in it, so equalise them
        // (both macros) and disable learning for a deterministic check.
        for i in 0..m.macros.len() {
            let w = m.macros[i].memory_block.hippo.hippocampus.snapshot();
            m2.macros[i]
                .memory_block
                .hippo
                .hippocampus
                .replace_weights(w);
        }
        m.hippo_learning = false;
        m2.hippo_learning = false;
        let a = m.forward_token(72, false);
        let b = m2.forward_token(72, false);
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "ctxb roundtrip logits err {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn model_loss_is_finite() {
        let mut m = micro_model();
        let tok = ByteBpeTokenizer::new();
        let seqs: Vec<Vec<u32>> = [
            "hello world",
            "rust is fast",
            "the cat sat",
            "one two three",
        ]
        .iter()
        .map(|s| tok.encode(s))
        .collect();
        let loss = m.loss_for_sequences(&seqs);
        assert!(loss.is_finite() && loss > 0.0, "loss {loss}");
    }
}
