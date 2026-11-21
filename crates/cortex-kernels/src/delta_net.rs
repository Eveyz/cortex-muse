//! Gated DeltaNet — the linear state-space layers 1–3 of Cortex-Local.
//!
//! # Mathematical contract (single head, fixed once here)
//!
//! State: `S ∈ R^{dv × dk}` (row-major `[dv][dk]`), `S_0 = 0`.
//! Per token `t`, with L2-normalised `k_t, q_t` (`q` additionally scaled
//! by `1/√dk`), value `v_t`, gate `β_t ∈ (0,1)`, decay `α_t ∈ (0,1]`:
//!
//! ```text
//! u_t     = S_{t-1} @ k_t            # what the memory currently predicts
//! δ_t     = v_t - α_t * u_t          # delta-rule error (note the α_t)
//! S_t     = α_t * S_{t-1} + β_t * δ_t ⊗ k_t
//!         = α_t * S_{t-1} @ (I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ
//! o_t     = S_t @ q_t                # output uses the *updated* state
//! ```
//!
//! This is the delta rule (gradient step on `½‖S k - v‖²` with step
//! `β_t`) plus the Gated DeltaNet decay `α_t`. It is **not** vanilla
//! linear attention (`S += v ⊗ k`): the `- β (S k) kᵀ` term erases the
//! old association for `k_t` before writing the new one, which is what
//! lets a fixed-size state overwrite stale facts instead of blurring
//! them — the property the hippocampus layer (layer 4) relies on.
//!
//! # Complexity
//!
//! * `step`: `O(dv·dk)` time, `O(dv·dk)` state, **`O(1)` in sequence
//!   length** — no KV cache, no per-token allocation in the hot path
//!   when `step_with_scratch` is used. State size is constant no matter
//!   how long the session runs.
//! * `forward_chunk` (prefill / training): chunkwise form below, same
//!   mathematics as running `step` token by token, verified by test.
//!
//! # Chunkwise form (WY / triangular solve)
//!
//! Inside a chunk of length `L`, write the (ungated) delta update as
//! `S_i = S_{i-1} + w_i ⊗ k_i` with `w_i = β_i (v_i - S_{i-1} k_i)`.
//! Then `p_i = S_{i-1} k_i = S_0 k_i + Σ_{j<i} w_j (k_j·k_i)`, so with
//! `e_i = v_i - S_0 k_i` and `A_ij = k_j·k_i (j < i)`:
//!
//! ```text
//! (I + diag(β) · A) w = diag(β) · e        # lower-triangular system
//! o_i = S_0 q_i + Σ_{j≤i} w_j (k_j·q_i)
//! S_L = S_0 + Σ_j w_j ⊗ k_j
//! ```
//!
//! Gating is reduced to the ungated case by the cumulative-decay trick:
//! with `Γ_i = Π_{s≤i} α_s`, define `S'_i = S_i / Γ_i` and
//! `v'_i = v_i / Γ_i`; then `S'` follows the ungated rule with the same
//! `k, β`, and `o_i = Γ_i · (S'_i q_i)`, `S_L = Γ_{L-1} · S'_L`.
//! If `Γ_{L-1}` gets too small (`< 1e-3`, i.e. the chunk would divide by
//! a near-zero number) the chunk falls back to the sequential `step`
//! loop — same result, just no chunk parallelism for that chunk.
//!
//! # References
//!
//! * Yang et al., "Parallelizing Linear Transformers with the Delta
//!   Rule over Sequence Length" (DeltaNet), NeurIPS 2024.
//! * Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta
//!   Rule" (Gated DeltaNet), arXiv:2412.06464.

use crate::bit158::BitLinearWeights;

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

pub use crate::ops::{sigmoid, softplus};

/// Gated DeltaNet decay: `α = exp(-exp(log_decay) * softplus(a))`.
/// `log_decay` is a per-head learned parameter (typically initialised so
/// `α ≈ 0.99`); `a` is the per-token logit from the α-projection.
#[inline(always)]
pub fn alpha_from_logit(a: f32, log_decay: f32) -> f32 {
    (-log_decay.exp() * softplus(a)).exp()
}

/// In-place L2 normalisation; a zero vector stays zero.
pub fn l2_normalize_inplace(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    let inv = 1.0 / (norm_sq.sqrt().max(1e-6));
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub use crate::ops::rms_norm;

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// 1. Single-head delta-rule state: O(1) memory recursion
// ---------------------------------------------------------------------------

/// Scratch buffers for the hot `step` path (no per-token allocation).
pub struct DeltaScratch {
    pub u: Vec<f32>,     // S @ k, length dv
    pub delta: Vec<f32>, // v - α·u, length dv
}

impl DeltaScratch {
    pub fn new(dv: usize) -> Self {
        Self {
            u: vec![0.0; dv],
            delta: vec![0.0; dv],
        }
    }
}

/// One head's recurrent state `S: [dv, dk]`, row-major.
#[derive(Clone, Debug)]
pub struct DeltaState {
    pub dk: usize,
    pub dv: usize,
    /// `s[v * dk + k]`.
    pub s: Vec<f32>,
}

impl DeltaState {
    pub fn new(dv: usize, dk: usize) -> Self {
        Self {
            dk,
            dv,
            s: vec![0.0; dv * dk],
        }
    }

    pub fn reset(&mut self) {
        self.s.fill(0.0);
    }

    /// State size in bytes — constant in sequence length.
    pub fn memory_bytes(&self) -> usize {
        self.s.len() * core::mem::size_of::<f32>()
    }

    /// Predicted value for a key under the current state: `S @ k`.
    pub fn predict(&self, k: &[f32], out: &mut [f32]) {
        debug_assert_eq!(k.len(), self.dk);
        debug_assert_eq!(out.len(), self.dv);
        for v in 0..self.dv {
            out[v] = dot(&self.s[v * self.dk..(v + 1) * self.dk], k);
        }
    }

    /// Single-token gated delta update + output. Allocates two small
    /// scratch vectors; use [`DeltaState::step_with_scratch`] on the
    /// hot path.
    pub fn step(
        &mut self,
        k: &[f32],
        q: &[f32],
        v: &[f32],
        beta: f32,
        alpha: f32,
        out: &mut [f32],
    ) {
        let mut scratch = DeltaScratch::new(self.dv);
        self.step_with_scratch(k, q, v, beta, alpha, out, &mut scratch);
    }

    /// Hot path: `S_t, o_t` from `S_{t-1}, k_t, q_t, v_t, β_t, α_t`.
    /// `k, q` must already be L2-normalised (and `q` scaled by `1/√dk`)
    /// by the caller — the layer does that, the core does not re-do it.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_scratch(
        &mut self,
        k: &[f32],
        q: &[f32],
        v: &[f32],
        beta: f32,
        alpha: f32,
        out: &mut [f32],
        scratch: &mut DeltaScratch,
    ) {
        debug_assert_eq!(k.len(), self.dk);
        debug_assert_eq!(q.len(), self.dk);
        debug_assert_eq!(v.len(), self.dv);
        debug_assert_eq!(out.len(), self.dv);
        debug_assert!((0.0..=1.0).contains(&beta), "beta must be in [0, 1]");
        debug_assert!((0.0..=1.0).contains(&alpha), "alpha must be in (0, 1]");
        let (dv, dk) = (self.dv, self.dk);

        // u = S @ k
        for r in 0..dv {
            scratch.u[r] = dot(&self.s[r * dk..(r + 1) * dk], k);
        }
        // δ = v - α·u
        for r in 0..dv {
            scratch.delta[r] = v[r] - alpha * scratch.u[r];
        }
        // S = α·S + β·δ ⊗ k   (in place, row by row)
        for r in 0..dv {
            let row = &mut self.s[r * dk..(r + 1) * dk];
            let d = beta * scratch.delta[r];
            for c in 0..dk {
                row[c] = alpha * row[c] + d * k[c];
            }
        }
        // o = S @ q (updated state)
        for r in 0..dv {
            out[r] = dot(&self.s[r * dk..(r + 1) * dk], q);
        }
    }

    /// Chunkwise forward over `chunk_len` tokens.
    ///
    /// Layouts (all row-major, one row per token):
    /// `ks, qs: [L][dk]`, `vs: [L][dv]`, `betas, alphas: [L]`,
    /// `outs: [L][dv]`. Updates `self` to the state after the chunk.
    /// Bit-identical in mathematics to calling `step` `L` times
    /// (up to float reassociation); see the module docs for the derivation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunk(
        &mut self,
        ks: &[f32],
        qs: &[f32],
        vs: &[f32],
        betas: &[f32],
        alphas: &[f32],
        chunk_len: usize,
        outs: &mut [f32],
    ) {
        let (dv, dk) = (self.dv, self.dk);
        let l = chunk_len;
        debug_assert_eq!(ks.len(), l * dk);
        debug_assert_eq!(qs.len(), l * dk);
        debug_assert_eq!(vs.len(), l * dv);
        debug_assert_eq!(betas.len(), l);
        debug_assert_eq!(alphas.len(), l);
        debug_assert_eq!(outs.len(), l * dv);
        if l == 0 {
            return;
        }
        if l == 1 {
            let mut o = vec![0.0; dv];
            self.step(
                &ks[0..dk],
                &qs[0..dk],
                &vs[0..dv],
                betas[0],
                alphas[0],
                &mut o,
            );
            outs[0..dv].copy_from_slice(&o);
            return;
        }

        // Cumulative decay Γ_i.
        let mut gamma = vec![0.0f32; l];
        let mut g = 1.0f32;
        for i in 0..l {
            g *= alphas[i];
            gamma[i] = g;
        }
        // Guard: dividing by a tiny Γ is numerically unsafe — fall back
        // to the sequential loop (same maths, no chunk parallelism).
        if gamma[l - 1] < 1e-3 {
            let mut scratch = DeltaScratch::new(dv);
            for i in 0..l {
                let (ko, qo, vo, oo) = (
                    &ks[i * dk..(i + 1) * dk],
                    &qs[i * dk..(i + 1) * dk],
                    &vs[i * dv..(i + 1) * dv],
                    &mut outs[i * dv..(i + 1) * dv],
                );
                // `step_with_scratch` needs &mut out separate from the
                // borrowed slices above; copy through a temp.
                let mut o = vec![0.0f32; dv];
                self.step_with_scratch(ko, qo, vo, betas[i], alphas[i], &mut o, &mut scratch);
                oo.copy_from_slice(&o);
            }
            return;
        }

        let s0: Vec<f32> = self.s.clone(); // state at chunk start

        // e_i = v_i / Γ_i - S_0 @ k_i
        let mut e = vec![0.0f32; l * dv];
        let mut pred = vec![0.0f32; dv];
        for i in 0..l {
            let k = &ks[i * dk..(i + 1) * dk];
            for r in 0..dv {
                pred[r] = dot(&s0[r * dk..(r + 1) * dk], k);
            }
            for r in 0..dv {
                e[i * dv + r] = vs[i * dv + r] / gamma[i] - pred[r];
            }
        }

        // Forward substitution for w: w_i = β_i·e_i - β_i·Σ_{j<i} (k_j·k_i) w_j
        let mut w = vec![0.0f32; l * dv];
        for i in 0..l {
            let ki = &ks[i * dk..(i + 1) * dk];
            for j in 0..i {
                let kj = &ks[j * dk..(j + 1) * dk];
                let aji = dot(kj, ki);
                if aji != 0.0 {
                    for r in 0..dv {
                        w[i * dv + r] -= betas[i] * aji * w[j * dv + r];
                    }
                }
            }
            for r in 0..dv {
                w[i * dv + r] += betas[i] * e[i * dv + r];
            }
        }

        // o'_i = S_0 q_i + Σ_{j≤i} w_j (k_j·q_i);  o_i = Γ_i · o'_i
        for i in 0..l {
            let qi = &qs[i * dk..(i + 1) * dk];
            let mut o = vec![0.0f32; dv];
            for r in 0..dv {
                o[r] = dot(&s0[r * dk..(r + 1) * dk], qi);
            }
            for j in 0..=i {
                let kj = &ks[j * dk..(j + 1) * dk];
                let kj_qi = dot(kj, qi);
                if kj_qi != 0.0 {
                    for r in 0..dv {
                        o[r] += w[j * dv + r] * kj_qi;
                    }
                }
            }
            for r in 0..dv {
                outs[i * dv + r] = gamma[i] * o[r];
            }
        }

        // S_L = Γ_{L-1} · (S_0 + Σ_j w_j ⊗ k_j)
        for r in 0..dv {
            for c in 0..dk {
                let mut acc = s0[r * dk + c];
                for j in 0..l {
                    acc += w[j * dv + r] * ks[j * dk + c];
                }
                self.s[r * dk + c] = gamma[l - 1] * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Deterministic ternary initialisation (toy / test weights, no rand dep)
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* — used only to build toy ternary weights for
/// tests and examples, never in the trained model path.
pub fn xorshift64star(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Random ternary matrix (`-1/0/+1`, ~1/3 each) in row-major order.
pub fn random_ternary_matrix(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
    let mut st = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..rows * cols)
        .map(|_| match xorshift64star(&mut st) % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect()
}

/// Public toy-projection constructor (used by `ops::BitSwiGLU::new_toy`
/// and by `cortex-ssm` / `cortex-runtime` toy models).
pub fn make_ternary_proj(rows: usize, cols: usize, seed: u64, scale: f32) -> BitLinearWeights {
    ternary_proj(rows, cols, seed, scale)
}

fn ternary_proj(rows: usize, cols: usize, seed: u64, scale: f32) -> BitLinearWeights {
    let w = random_ternary_matrix(rows, cols, seed);
    BitLinearWeights::from_ternary_matrix(&w, rows, cols, &vec![scale; rows])
}

// ---------------------------------------------------------------------------
// 3. One Gated DeltaNet layer (multi-head, Bit158 projections)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GatedDeltaNetConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    /// Prefill chunk size for `forward_chunk` (default 64).
    pub chunk_size: usize,
}

impl GatedDeltaNetConfig {
    pub fn new(d_model: usize, num_heads: usize, head_dim_k: usize, head_dim_v: usize) -> Self {
        Self {
            d_model,
            num_heads,
            head_dim_k,
            head_dim_v,
            chunk_size: 64,
        }
    }
    pub fn total_k(&self) -> usize {
        self.num_heads * self.head_dim_k
    }
    pub fn total_v(&self) -> usize {
        self.num_heads * self.head_dim_v
    }
}

/// One layer of the frozen 1.58-bit cognitive core (layers 1–3).
///
/// Per token: `h = RMSNorm(x)`; `q, k, v, β-logits, α-logits` come from
/// five `BitLinearWeights` projections; each head runs its own gated
/// delta rule on its own `DeltaState`; head outputs are concatenated
/// and passed through the output projection, with a residual:
/// `y = x + W_o @ concat(head_outs)`.
///
/// The layer owns no heap allocation on the step path beyond the
/// scratch buffers created once in the constructor.
pub struct GatedDeltaNetLayer {
    pub cfg: GatedDeltaNetConfig,
    pub rms_weight: Vec<f32>,
    pub q_proj: BitLinearWeights,
    pub k_proj: BitLinearWeights,
    pub v_proj: BitLinearWeights,
    pub beta_proj: BitLinearWeights,
    pub alpha_proj: BitLinearWeights,
    pub out_proj: BitLinearWeights,
    /// Per-head `log_decay` for `alpha_from_logit` (α ≈ 0.99 at init).
    pub log_decay: Vec<f32>,
    pub states: Vec<DeltaState>,
    // Scratch, reused every step (sizes fixed at construction).
    scratch_h: Vec<f32>,
    scratch_q: Vec<f32>,
    scratch_k: Vec<f32>,
    scratch_v: Vec<f32>,
    scratch_beta: Vec<f32>,
    scratch_alpha: Vec<f32>,
    scratch_heads: Vec<f32>,
    scratch_out: Vec<f32>,
    delta_scratch: Vec<DeltaScratch>,
}

impl GatedDeltaNetLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: GatedDeltaNetConfig,
        rms_weight: Vec<f32>,
        q_proj: BitLinearWeights,
        k_proj: BitLinearWeights,
        v_proj: BitLinearWeights,
        beta_proj: BitLinearWeights,
        alpha_proj: BitLinearWeights,
        out_proj: BitLinearWeights,
        log_decay: Vec<f32>,
    ) -> Self {
        assert_eq!(rms_weight.len(), cfg.d_model);
        assert_eq!(q_proj.rows, cfg.total_k());
        assert_eq!(k_proj.rows, cfg.total_k());
        assert_eq!(v_proj.rows, cfg.total_v());
        assert_eq!(beta_proj.rows, cfg.num_heads);
        assert_eq!(alpha_proj.rows, cfg.num_heads);
        assert_eq!(out_proj.rows, cfg.d_model);
        assert_eq!(out_proj.cols, cfg.total_v());
        assert_eq!(log_decay.len(), cfg.num_heads);
        let states = (0..cfg.num_heads)
            .map(|_| DeltaState::new(cfg.head_dim_v, cfg.head_dim_k))
            .collect();
        let delta_scratch = (0..cfg.num_heads)
            .map(|_| DeltaScratch::new(cfg.head_dim_v))
            .collect();
        Self {
            scratch_h: vec![0.0; cfg.d_model],
            scratch_q: vec![0.0; cfg.total_k()],
            scratch_k: vec![0.0; cfg.total_k()],
            scratch_v: vec![0.0; cfg.total_v()],
            scratch_beta: vec![0.0; cfg.num_heads],
            scratch_alpha: vec![0.0; cfg.num_heads],
            scratch_heads: vec![0.0; cfg.total_v()],
            scratch_out: vec![0.0; cfg.d_model],
            states,
            delta_scratch,
            cfg,
            rms_weight,
            q_proj,
            k_proj,
            v_proj,
            beta_proj,
            alpha_proj,
            out_proj,
            log_decay,
        }
    }

    /// Toy constructor: deterministic ternary projections, for tests,
    /// the one-shot sandbox and the runtime example. Not a trained model.
    pub fn new_toy(cfg: GatedDeltaNetConfig, seed: u64) -> Self {
        let d = cfg.d_model;
        let scale = (1.0 / d as f32).sqrt() * 0.5;
        // log_decay chosen so α ≈ 0.99 when the α-logit is ~0:
        // α = exp(-exp(ld)·softplus(0)) = exp(-exp(ld)·0.693) = 0.99
        // → exp(ld) ≈ 0.0145 → ld ≈ -4.23.
        let log_decay = vec![-4.23f32; cfg.num_heads];
        Self::new(
            cfg.clone(),
            vec![1.0; d],
            ternary_proj(cfg.total_k(), d, seed + 1, scale),
            ternary_proj(cfg.total_k(), d, seed + 2, scale),
            ternary_proj(cfg.total_v(), d, seed + 3, scale),
            ternary_proj(cfg.num_heads, d, seed + 4, 0.1),
            ternary_proj(cfg.num_heads, d, seed + 5, 0.1),
            ternary_proj(d, cfg.total_v(), seed + 6, scale),
            log_decay,
        )
    }

    pub fn reset_states(&mut self) {
        for s in self.states.iter_mut() {
            s.reset();
        }
    }

    /// Total recurrent state in bytes — O(1) in sequence length.
    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(|s| s.memory_bytes()).sum()
    }

    // RMSNorm is a free function (not a `&self` method) so callers can
    // pass `&self.rms_weight` and `&mut self.scratch_h` as disjoint
    // field borrows without a borrow-checker conflict.

    /// Project one normalised token into per-head q/k/v/β/α.
    /// `q, k` leave here L2-normalised, `q` scaled by `1/√dk`.
    fn project_token(&mut self) {
        // Clone the projections' inputs first: `forward_gemv` borrows
        // `self.q_proj` immutably while `scratch_h` is borrowed mutably
        // through `self` — split the borrows explicitly.
        let h = self.scratch_h.clone();
        self.q_proj.forward_gemv(&h, &mut self.scratch_q);
        self.k_proj.forward_gemv(&h, &mut self.scratch_k);
        self.v_proj.forward_gemv(&h, &mut self.scratch_v);
        self.beta_proj.forward_gemv(&h, &mut self.scratch_beta);
        self.alpha_proj.forward_gemv(&h, &mut self.scratch_alpha);
        let dk = self.cfg.head_dim_k;
        let q_scale = 1.0 / (dk as f32).sqrt();
        for head in 0..self.cfg.num_heads {
            let qs = &mut self.scratch_q[head * dk..(head + 1) * dk];
            l2_normalize_inplace(qs);
            for v in qs.iter_mut() {
                *v *= q_scale;
            }
            let ks = &mut self.scratch_k[head * dk..(head + 1) * dk];
            l2_normalize_inplace(ks);
        }
    }

    /// One streaming token: `y = x + W_o @ Delta(RMSNorm(x))`.
    /// Advances every head's state by exactly one step.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cfg.d_model);
        assert_eq!(out.len(), self.cfg.d_model);
        {
            // Disjoint field borrows: rms_weight (shared) + scratch_h (mut).
            let w = &self.rms_weight;
            let h = &mut self.scratch_h;
            rms_norm(w, x, h);
        }
        self.project_token();
        // Per-head delta step.
        let (dk, dv) = (self.cfg.head_dim_k, self.cfg.head_dim_v);
        for head in 0..self.cfg.num_heads {
            let (k, q, v) = (
                self.scratch_k[head * dk..(head + 1) * dk].to_vec(),
                self.scratch_q[head * dk..(head + 1) * dk].to_vec(),
                self.scratch_v[head * dv..(head + 1) * dv].to_vec(),
            );
            let beta = sigmoid(self.scratch_beta[head]);
            let alpha = alpha_from_logit(self.scratch_alpha[head], self.log_decay[head]);
            let mut o = vec![0.0f32; dv];
            // Split borrows: state + its scratch live in different fields.
            let (states, scratches) = (&mut self.states, &mut self.delta_scratch);
            states[head].step_with_scratch(&k, &q, &v, beta, alpha, &mut o, &mut scratches[head]);
            self.scratch_heads[head * dv..(head + 1) * dv].copy_from_slice(&o);
        }
        let heads = self.scratch_heads.clone();
        self.out_proj.forward_gemv(&heads, &mut self.scratch_out);
        for i in 0..self.cfg.d_model {
            out[i] = x[i] + self.scratch_out[i];
        }
    }

    /// Prefill / training forward over `xs: [L][d_model]` → `outs: [L][d_model]`.
    /// Uses the chunkwise delta rule per head; must match `forward_step`
    /// called `L` times (verified in tests). Advances the states by `L`.
    pub fn forward_chunk(&mut self, xs: &[f32], outs: &mut [f32]) {
        let d = self.cfg.d_model;
        let l = xs.len() / d;
        assert_eq!(xs.len(), l * d);
        assert_eq!(outs.len(), l * d);
        let chunk = self.cfg.chunk_size.max(1);
        let mut start = 0;
        while start < l {
            let end = (start + chunk).min(l);
            self.forward_chunk_one(
                &xs[start * d..end * d],
                &mut outs[start * d..end * d],
                end - start,
            );
            start = end;
        }
    }

    fn forward_chunk_one(&mut self, xs: &[f32], outs: &mut [f32], l: usize) {
        let d = self.cfg.d_model;
        let (dk, dv, nh) = (self.cfg.head_dim_k, self.cfg.head_dim_v, self.cfg.num_heads);
        // Project every token in the chunk.
        let mut q_all = vec![0.0f32; l * nh * dk];
        let mut k_all = vec![0.0f32; l * nh * dk];
        let mut v_all = vec![0.0f32; l * nh * dv];
        let mut beta_all = vec![0.0f32; l * nh];
        let mut alpha_all = vec![0.0f32; l * nh];
        let mut h = vec![0.0f32; d];
        for t in 0..l {
            rms_norm(&self.rms_weight, &xs[t * d..(t + 1) * d], &mut h);
            let mut q = vec![0.0; nh * dk];
            let mut k = vec![0.0; nh * dk];
            let mut v = vec![0.0; nh * dv];
            let mut b = vec![0.0; nh];
            let mut a = vec![0.0; nh];
            self.q_proj.forward_gemv(&h, &mut q);
            self.k_proj.forward_gemv(&h, &mut k);
            self.v_proj.forward_gemv(&h, &mut v);
            self.beta_proj.forward_gemv(&h, &mut b);
            self.alpha_proj.forward_gemv(&h, &mut a);
            let q_scale = 1.0 / (dk as f32).sqrt();
            for head in 0..nh {
                let qs = &mut q[head * dk..(head + 1) * dk];
                l2_normalize_inplace(qs);
                for val in qs.iter_mut() {
                    *val *= q_scale;
                }
                l2_normalize_inplace(&mut k[head * dk..(head + 1) * dk]);
                beta_all[t * nh + head] = sigmoid(b[head]);
                alpha_all[t * nh + head] = alpha_from_logit(a[head], self.log_decay[head]);
            }
            q_all[t * nh * dk..(t + 1) * nh * dk].copy_from_slice(&q);
            k_all[t * nh * dk..(t + 1) * nh * dk].copy_from_slice(&k);
            v_all[t * nh * dv..(t + 1) * nh * dv].copy_from_slice(&v);
        }
        // Per-head chunkwise delta, then per-token output projection.
        let mut heads_all = vec![0.0f32; l * nh * dv];
        for head in 0..nh {
            // Gather this head's rows into contiguous [L][dk] / [L][dv].
            let mut ks = vec![0.0f32; l * dk];
            let mut qs = vec![0.0f32; l * dk];
            let mut vs = vec![0.0f32; l * dv];
            let mut betas = vec![0.0f32; l];
            let mut alphas = vec![0.0f32; l];
            for t in 0..l {
                ks[t * dk..(t + 1) * dk]
                    .copy_from_slice(&k_all[(t * nh + head) * dk..(t * nh + head + 1) * dk]);
                qs[t * dk..(t + 1) * dk]
                    .copy_from_slice(&q_all[(t * nh + head) * dk..(t * nh + head + 1) * dk]);
                vs[t * dv..(t + 1) * dv]
                    .copy_from_slice(&v_all[(t * nh + head) * dv..(t * nh + head + 1) * dv]);
                betas[t] = beta_all[t * nh + head];
                alphas[t] = alpha_all[t * nh + head];
            }
            let mut head_outs = vec![0.0f32; l * dv];
            self.states[head].forward_chunk(&ks, &qs, &vs, &betas, &alphas, l, &mut head_outs);
            for t in 0..l {
                heads_all[(t * nh + head) * dv..(t * nh + head + 1) * dv]
                    .copy_from_slice(&head_outs[t * dv..(t + 1) * dv]);
            }
        }
        for t in 0..l {
            let mut o = vec![0.0f32; d];
            self.out_proj
                .forward_gemv(&heads_all[t * nh * dv..(t + 1) * nh * dv], &mut o);
            for i in 0..d {
                outs[t * d + i] = xs[t * d + i] + o[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The three SSM layers as one stack (layers 1–3 of the core)
// ---------------------------------------------------------------------------

/// Layers 1–3 stacked: `x -> L1 -> L2 -> L3`, each a residual
/// Gated DeltaNet layer with its own state. Layer 4 (hippocampus) sits
/// on top of this stack in `cortex-hippocampus` / `cortex-runtime`.
pub struct GatedDeltaNetStack {
    pub layers: Vec<GatedDeltaNetLayer>,
}

impl GatedDeltaNetStack {
    pub fn new(layers: Vec<GatedDeltaNetLayer>) -> Self {
        assert!(!layers.is_empty(), "stack needs at least one layer");
        Self { layers }
    }

    pub fn new_toy(cfg: GatedDeltaNetConfig, seed: u64) -> Self {
        let layers = (0..3)
            .map(|i| GatedDeltaNetLayer::new_toy(cfg.clone(), seed + 1000 * i as u64))
            .collect();
        Self::new(layers)
    }

    pub fn reset_states(&mut self) {
        for l in self.layers.iter_mut() {
            l.reset_states();
        }
    }

    pub fn state_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.state_bytes()).sum()
    }

    pub fn d_model(&self) -> usize {
        self.layers[0].cfg.d_model
    }

    /// Streaming step through all three layers.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) {
        let d = self.d_model();
        let mut cur = x.to_vec();
        let mut nxt = vec![0.0f32; d];
        for layer in self.layers.iter_mut() {
            layer.forward_step(&cur, &mut nxt);
            core::mem::swap(&mut cur, &mut nxt);
        }
        out.copy_from_slice(&cur);
    }

    /// Prefill over `[L][d_model]`; must match repeated `forward_step`.
    pub fn forward_chunk(&mut self, xs: &[f32], outs: &mut [f32]) {
        let d = self.d_model();
        let l = xs.len() / d;
        let mut cur = xs.to_vec();
        let mut nxt = vec![0.0f32; l * d];
        for layer in self.layers.iter_mut() {
            layer.forward_chunk(&cur, &mut nxt);
            core::mem::swap(&mut cur, &mut nxt);
        }
        outs.copy_from_slice(&cur);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed ^ 0x1234_5678_9ABC_DEF0;
        (0..n)
            .map(|_| {
                let u = (xorshift64star(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn normalised(v: &mut [f32]) {
        l2_normalize_inplace(v);
    }

    #[test]
    fn delta_step_binds_and_recalls() {
        // Teach k -> v once with β=1, α=1, then query with the same k.
        let (dk, dv) = (8, 8);
        let mut s = DeltaState::new(dv, dk);
        let mut k = rand_vec(dk, 1);
        normalised(&mut k);
        let v = rand_vec(dv, 2);
        let q = k.clone();
        // q is scaled in the layer; here use it raw for the core test.
        let mut out = vec![0.0; dv];
        s.step(&k, &q, &v, 1.0, 1.0, &mut out);
        // After one β=1 write, S = v ⊗ k, so S @ k = v·‖k‖² = v.
        for r in 0..dv {
            assert!((out[r] - v[r]).abs() < 1e-5, "recall failed at {r}");
        }
    }

    #[test]
    fn delta_step_overwrites_stale_fact() {
        // Same key, new value: the delta term must erase the old one.
        let (dk, dv) = (8, 8);
        let mut s = DeltaState::new(dv, dk);
        let mut k = rand_vec(dk, 3);
        normalised(&mut k);
        let v1 = rand_vec(dv, 4);
        let v2 = rand_vec(dv, 5);
        let mut out = vec![0.0; dv];
        s.step(&k, &k, &v1, 1.0, 1.0, &mut out);
        s.step(&k, &k, &v2, 1.0, 1.0, &mut out);
        for r in 0..dv {
            assert!((out[r] - v2[r]).abs() < 1e-5, "overwrite failed at {r}");
        }
        // A different (near-orthogonal) key must not return v2.
        let mut k2 = rand_vec(dk, 6);
        normalised(&mut k2);
        // Make k2 orthogonal-ish to k by construction is hard with random
        // vectors; just check the output is not exactly v2.
        s.step(&k2, &k2, &vec![0.0; dv], 0.0, 1.0, &mut out);
        let err: f32 = out
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(err > 1e-3, "unrelated key should not recall v2 exactly");
    }

    #[test]
    fn delta_step_decay_clears_state() {
        let (dk, dv) = (8, 8);
        let mut s = DeltaState::new(dv, dk);
        let mut k = rand_vec(dk, 7);
        normalised(&mut k);
        let v = rand_vec(dv, 8);
        let mut out = vec![0.0; dv];
        s.step(&k, &k, &v, 1.0, 1.0, &mut out);
        // α=0, β=0: state is wiped, output is 0.
        s.step(&k, &k, &vec![0.0; dv], 0.0, 0.0, &mut out);
        assert!(out.iter().all(|&x| x.abs() < 1e-6));
        assert!(s.s.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn chunkwise_matches_sequential() {
        for (dk, dv, l) in [
            (8usize, 8usize, 1usize),
            (8, 8, 5),
            (16, 8, 12),
            (8, 16, 16),
        ] {
            let mut ks = rand_vec(l * dk, 10);
            let mut qs = rand_vec(l * dk, 11);
            let vs = rand_vec(l * dv, 12);
            for t in 0..l {
                normalised(&mut ks[t * dk..(t + 1) * dk]);
                normalised(&mut qs[t * dk..(t + 1) * dk]);
            }
            // α close to 1 (the realistic regime) and β in (0, 1).
            let betas: Vec<f32> = (0..l).map(|i| 0.3 + 0.1 * ((i * 7) % 5) as f32).collect();
            let alphas: Vec<f32> = (0..l)
                .map(|i| 0.99 - 0.005 * ((i * 3) % 4) as f32)
                .collect();

            // Sequential reference.
            let mut s_seq = DeltaState::new(dv, dk);
            let mut out_seq = vec![0.0f32; l * dv];
            for t in 0..l {
                s_seq.step(
                    &ks[t * dk..(t + 1) * dk],
                    &qs[t * dk..(t + 1) * dk],
                    &vs[t * dv..(t + 1) * dv],
                    betas[t],
                    alphas[t],
                    &mut out_seq[t * dv..(t + 1) * dv],
                );
            }
            // Chunkwise.
            let mut s_chunk = DeltaState::new(dv, dk);
            let mut out_chunk = vec![0.0f32; l * dv];
            s_chunk.forward_chunk(&ks, &qs, &vs, &betas, &alphas, l, &mut out_chunk);

            let max_out_err = out_seq
                .iter()
                .zip(out_chunk.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_state_err = s_seq
                .s
                .iter()
                .zip(s_chunk.s.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_out_err < 1e-4,
                "dk={dk} dv={dv} L={l}: out err {max_out_err}"
            );
            assert!(
                max_state_err < 1e-4,
                "dk={dk} dv={dv} L={l}: state err {max_state_err}"
            );
        }
    }

    #[test]
    fn chunkwise_fallback_small_alpha_matches_sequential() {
        // α = 0.5 per token, L = 16 → Γ ≈ 1.5e-5 < 1e-3 triggers fallback.
        let (dk, dv, l) = (8usize, 8usize, 16usize);
        let mut ks = rand_vec(l * dk, 20);
        let mut qs = rand_vec(l * dk, 21);
        let vs = rand_vec(l * dv, 22);
        for t in 0..l {
            normalised(&mut ks[t * dk..(t + 1) * dk]);
            normalised(&mut qs[t * dk..(t + 1) * dk]);
        }
        let betas = vec![0.5f32; l];
        let alphas = vec![0.5f32; l];
        let mut s_seq = DeltaState::new(dv, dk);
        let mut out_seq = vec![0.0f32; l * dv];
        for t in 0..l {
            s_seq.step(
                &ks[t * dk..(t + 1) * dk],
                &qs[t * dk..(t + 1) * dk],
                &vs[t * dv..(t + 1) * dv],
                betas[t],
                alphas[t],
                &mut out_seq[t * dv..(t + 1) * dv],
            );
        }
        let mut s_chunk = DeltaState::new(dv, dk);
        let mut out_chunk = vec![0.0f32; l * dv];
        s_chunk.forward_chunk(&ks, &qs, &vs, &betas, &alphas, l, &mut out_chunk);
        let err = out_seq
            .iter()
            .zip(out_chunk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-4, "fallback chunk err {err}");
    }

    #[test]
    fn layer_chunk_matches_step_and_state_is_o1() {
        let cfg = GatedDeltaNetConfig {
            chunk_size: 4,
            ..GatedDeltaNetConfig::new(32, 2, 8, 8)
        };
        let mut layer_step = GatedDeltaNetLayer::new_toy(cfg.clone(), 42);
        let mut layer_chunk = GatedDeltaNetLayer::new_toy(cfg.clone(), 42);
        let l = 10usize;
        let xs = rand_vec(l * 32, 99);
        // Streaming.
        let mut out_step = vec![0.0f32; l * 32];
        for t in 0..l {
            layer_step.forward_step(
                &xs[t * 32..(t + 1) * 32],
                &mut out_step[t * 32..(t + 1) * 32],
            );
        }
        // Prefill.
        let mut out_chunk = vec![0.0f32; l * 32];
        layer_chunk.forward_chunk(&xs, &mut out_chunk);
        let err = out_step
            .iter()
            .zip(out_chunk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "layer chunk vs step err {err}");

        // O(1) state: same bytes after 1 token and after 1000 tokens.
        let bytes_before = layer_step.state_bytes();
        let x = rand_vec(32, 7);
        let mut o = vec![0.0; 32];
        for _ in 0..1000 {
            layer_step.forward_step(&x, &mut o);
        }
        assert_eq!(layer_step.state_bytes(), bytes_before);
        assert!(
            o.iter().all(|v| v.is_finite()),
            "outputs must stay finite after 1000 steps"
        );
    }

    #[test]
    fn stack_chunk_matches_step() {
        let cfg = GatedDeltaNetConfig {
            chunk_size: 8,
            ..GatedDeltaNetConfig::new(32, 2, 8, 8)
        };
        let mut s_step = GatedDeltaNetStack::new_toy(cfg.clone(), 123);
        let mut s_chunk = GatedDeltaNetStack::new_toy(cfg.clone(), 123);
        let l = 9usize;
        let xs = rand_vec(l * 32, 55);
        let mut out_step = vec![0.0f32; l * 32];
        for t in 0..l {
            s_step.forward_step(
                &xs[t * 32..(t + 1) * 32],
                &mut out_step[t * 32..(t + 1) * 32],
            );
        }
        let mut out_chunk = vec![0.0f32; l * 32];
        s_chunk.forward_chunk(&xs, &mut out_chunk);
        let err = out_step
            .iter()
            .zip(out_chunk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-3, "stack chunk vs step err {err}");
        // 3 layers × 2 heads × 8×8 state × 4 bytes = 1536 bytes, constant.
        assert_eq!(s_step.state_bytes(), 3 * 2 * 8 * 8 * 4);
    }

    #[test]
    fn gates_are_in_range() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        let a = alpha_from_logit(0.0, -4.23);
        assert!(
            (0.98..1.0).contains(&a),
            "alpha at init should be ~0.99, got {a}"
        );
        assert!(alpha_from_logit(10.0, 0.0) < 0.01);
    }
}
