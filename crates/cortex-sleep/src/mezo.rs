//! `MezoScaleTuner` — MeZO zero-order tuning of the continuous
//! parameters `φ` only (Phase 4).
//!
//! `φ` is the concatenation of every `BitLinearWeights::scales`
//! (row scales) and every `rms_weight` vector in the model. All
//! discrete 2-bit weights stay frozen. No autograd graph is built:
//! the tuner only ever runs pure forward passes through a caller
//! supplied loss closure.
//!
//! One step, with deterministic seed `s` (the same `z ~ N(0, I)` is
//! regenerated from `s` three times, so `z` itself is never stored
//! and `φ` is perturbed in place — no extra copy of `φ`):
//! 1. `φ <- φ + εz`, `L+ = loss(φ)`;
//! 2. `φ <- φ - 2εz`, `L- = loss(φ)`;
//! 3. `g = (L+ - L-) / (2ε)`; `φ <- φ + εz - η·g·z`
//!    (the `+εz` restores `φ` to its pre-step value, then the
//!    projected-gradient step is applied).
//!
//! The loss closure receives `&[f32]` (the current `φ`) and returns
//! a scalar loss. In the runtime this closure sets the model's `φ`,
//! runs a forward pass over a mixed replay batch, and returns the
//! cross-entropy — so "no backward activation graph is allocated"
//! holds by construction: the closure only does forward passes.

use cortex_kernels::bit158::BitLinearWeights;

/// Deterministic PRNG (xorshift64*) with Box–Muller normals.
pub struct MezoRng {
    state: u64,
    spare: Option<f32>,
}

impl MezoRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub(crate) fn next_uniform(&mut self) -> f32 {
        // Uniform in (0, 1].
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (1.0 - u) as f32
    }

    /// Standard normal via Box–Muller (one spare cached).
    pub fn next_normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_uniform().max(1e-7);
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[derive(Clone, Debug)]
pub struct MezoStepReport {
    pub loss_plus: f32,
    pub loss_minus: f32,
    pub g_proj: f32,
}

#[derive(Clone, Debug)]
pub struct MezoScaleTuner {
    /// Perturbation size ε.
    pub eps: f32,
    /// Learning rate η.
    pub lr: f32,
    /// Base seed; step `t` uses seed `base_seed + t` so every step's
    /// perturbation direction is reproducible.
    pub base_seed: u64,
    pub steps_done: u64,
    /// Optional trust region: if `Some(f)`, no element moves by more
    /// than `f * |φ_i|` in one step. `None` (default) is the exact
    /// spec formula. A trust region is needed in practice because
    /// `φ` (row scales, ~0.03) is small while `g_proj` is O(1–10):
    /// with `lr = 0.2` one step moves a scale by ~1.2 (40x its value)
    /// and the model collapses to uniform logits; with `lr = 0.004`
    /// it diverges instead. The clip keeps sleep-time tuning safe.
    pub max_rel_step: Option<f32>,
}

impl Default for MezoScaleTuner {
    fn default() -> Self {
        Self {
            eps: 1e-2,
            lr: 0.05,
            base_seed: 0x00C0_FFEE,
            steps_done: 0,
            max_rel_step: None,
        }
    }
}

impl MezoScaleTuner {
    pub fn new(eps: f32, lr: f32, base_seed: u64) -> Self {
        Self {
            eps,
            lr,
            base_seed,
            steps_done: 0,
            max_rel_step: None,
        }
    }

    pub fn with_trust_region(mut self, frac: f32) -> Self {
        self.max_rel_step = Some(frac);
        self
    }

    /// One MeZO step on `phi` in place, averaging `n_dirs` independent
    /// perturbation directions. Each direction costs two forward loss
    /// evaluations; the directions are combined as
    /// `φ <- φ - η · (1/K) Σ_d g_d · z_d`.
    ///
    /// Why this exists: a single random direction in a 24k-dim `φ`
    /// captures only ~`1/√dim` of the gradient, so 30 single-direction
    /// steps move the loss by only ~3% on the toy model (measured),
    /// while a larger `lr` diverges. Averaging `K` directions reduces
    /// the gradient-estimate variance ~`K`-fold and lets the same 30
    /// steps descend significantly. `φ` is perturbed in place and
    /// restored by re-applying the regenerated `z` (no extra copy of
    /// `φ`); only the accumulated update vector (one `φ`-sized buffer)
    /// is allocated.
    pub fn step_multi(
        &mut self,
        phi: &mut [f32],
        loss: &mut dyn FnMut(&[f32]) -> f32,
        n_dirs: usize,
    ) -> Vec<MezoStepReport> {
        assert!(n_dirs >= 1);
        let mut update = vec![0.0f32; phi.len()];
        let mut reports = Vec::with_capacity(n_dirs);
        for d in 0..n_dirs {
            let seed = self
                .base_seed
                .wrapping_add(self.steps_done * 10_000 + d as u64);
            let eps = self.eps;
            {
                let mut rng = MezoRng::new(seed);
                for v in phi.iter_mut() {
                    *v += eps * rng.next_normal();
                }
            }
            let loss_plus = loss(phi);
            {
                let mut rng = MezoRng::new(seed);
                for v in phi.iter_mut() {
                    *v -= 2.0 * eps * rng.next_normal();
                }
            }
            let loss_minus = loss(phi);
            let g_proj = (loss_plus - loss_minus) / (2.0 * eps);
            // Restore φ to its pre-step value, and accumulate the update.
            {
                let mut rng = MezoRng::new(seed);
                for (i, v) in phi.iter_mut().enumerate() {
                    let z = rng.next_normal();
                    *v += eps * z;
                    update[i] -= self.lr * g_proj * z / n_dirs as f32;
                }
            }
            reports.push(MezoStepReport {
                loss_plus,
                loss_minus,
                g_proj,
            });
        }
        // Apply the averaged update (optionally trust-region clipped).
        for (i, v) in phi.iter_mut().enumerate() {
            let mut step = update[i];
            if let Some(frac) = self.max_rel_step {
                let cap = frac * v.abs().max(1e-6);
                step = step.clamp(-cap, cap);
            }
            *v += step;
        }
        self.steps_done += 1;
        reports
    }

    /// One MeZO step on `phi` in place. `loss` is called exactly twice.
    pub fn step(&mut self, phi: &mut [f32], loss: &mut dyn FnMut(&[f32]) -> f32) -> MezoStepReport {
        let seed = self.base_seed.wrapping_add(self.steps_done);
        let eps = self.eps;
        // Pass 1: φ + εz
        {
            let mut rng = MezoRng::new(seed);
            for v in phi.iter_mut() {
                *v += eps * rng.next_normal();
            }
        }
        let loss_plus = loss(phi);
        // Pass 2: φ - 2εz  (from φ+εz to φ-εz)
        {
            let mut rng = MezoRng::new(seed);
            for v in phi.iter_mut() {
                *v -= 2.0 * eps * rng.next_normal();
            }
        }
        let loss_minus = loss(phi);
        let g_proj = (loss_plus - loss_minus) / (2.0 * eps);
        // Pass 3: restore φ and apply η·g·z (optionally clipped to
        // the trust region).
        {
            let mut rng = MezoRng::new(seed);
            for v in phi.iter_mut() {
                let z = rng.next_normal();
                let mut step = eps * z - self.lr * g_proj * z;
                if let Some(frac) = self.max_rel_step {
                    let cap = frac * v.abs().max(1e-6);
                    step = step.clamp(-cap, cap);
                }
                *v += step;
            }
        }
        self.steps_done += 1;
        MezoStepReport {
            loss_plus,
            loss_minus,
            g_proj,
        }
    }
}

/// Collect `φ` from a model's continuous parameters: every row-scale
/// vector of every `BitLinearWeights`, then every RMS weight vector.
pub fn collect_phi(projections: &[&BitLinearWeights], rms_weights: &[&[f32]]) -> Vec<f32> {
    let mut phi = Vec::new();
    for p in projections {
        phi.extend_from_slice(&p.scales);
    }
    for r in rms_weights {
        phi.extend_from_slice(r);
    }
    phi
}

/// Scatter `φ` back into the same parameters, in the same order as
/// [`collect_phi`]. Panics on length mismatch.
pub fn scatter_phi(
    phi: &[f32],
    projections: &mut [&mut BitLinearWeights],
    rms_weights: &mut [&mut Vec<f32>],
) {
    let mut off = 0usize;
    for p in projections.iter_mut() {
        let n = p.scales.len();
        p.scales.copy_from_slice(&phi[off..off + n]);
        off += n;
    }
    for r in rms_weights.iter_mut() {
        let n = r.len();
        r.copy_from_slice(&phi[off..off + n]);
        off += n;
    }
    assert_eq!(off, phi.len(), "phi length mismatch on scatter");
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_kernels::make_ternary_proj;

    #[test]
    fn test_mezo_loss_descent() {
        // Quadratic loss in φ: a toy BitLinear layer's row scales are
        // φ; the loss is the MSE between the layer output and a fixed
        // target. 30 MeZO steps must clearly descend, using only
        // forward evaluations (the closure below does no backward pass
        // and allocates no activation graph — it is a plain GEMV).
        let d = 16usize;
        let rows = 8usize;
        let proj = make_ternary_proj(rows, d, 3, 0.25);
        let x: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.4).sin()).collect();
        // Target: output of the same layer with scales * 1.5.
        let mut target = vec![0.0f32; rows];
        {
            let mut p2 = proj.clone();
            for s in p2.scales.iter_mut() {
                *s *= 1.5;
            }
            p2.forward_gemv(&x, &mut target);
        }
        let mut phi = proj.scales.clone();
        let loss_of = |phi: &[f32]| -> f32 {
            let mut p = proj.clone();
            p.scales.copy_from_slice(phi);
            let mut out = vec![0.0f32; rows];
            p.forward_gemv(&x, &mut out);
            out.iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / rows as f32
        };
        let loss0 = loss_of(&phi);
        let mut tuner = MezoScaleTuner::new(0.01, 0.05, 1234);
        let mut loss_calls = 0usize;
        for _ in 0..30 {
            let mut loss = |phi: &[f32]| -> f32 {
                loss_calls += 1;
                loss_of(phi)
            };
            tuner.step(&mut phi, &mut loss);
        }
        assert_eq!(loss_calls, 60, "MeZO must call loss exactly twice per step");
        let loss1 = loss_of(&phi);
        assert!(
            loss1 < loss0 * 0.5,
            "MeZO did not descend: {loss0:.5} -> {loss1:.5}"
        );
        assert!(phi.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mezo_phi_roundtrip() {
        let p1 = make_ternary_proj(4, 8, 1, 0.2);
        let p2 = make_ternary_proj(6, 8, 2, 0.3);
        let rms = vec![1.0f32, 2.0, 3.0];
        let phi = collect_phi(&[&p1, &p2], &[&rms]);
        assert_eq!(phi.len(), 4 + 6 + 3);
        let mut q1 = p1.clone();
        let mut q2 = p2.clone();
        let mut qrms = rms.clone();
        let mut phi2 = phi.clone();
        for v in phi2.iter_mut() {
            *v += 1.0;
        }
        scatter_phi(&phi2, &mut [&mut q1, &mut q2], &mut [&mut qrms]);
        assert!(q1.scales.iter().all(|&s| (s - 1.2).abs() < 1e-6));
        assert_eq!(qrms, vec![2.0, 3.0, 4.0]);
    }
}
