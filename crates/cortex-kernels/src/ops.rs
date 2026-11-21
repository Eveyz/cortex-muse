//! Small operators shared by every layer: RMSNorm, activations, SwiGLU.

use crate::bit158::BitLinearWeights;

/// RMSNorm: `out[i] = x[i] / RMS(x) * weight[i]`, `eps = 1e-5`.
pub fn rms_norm(weight: &[f32], x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weight.len(), x.len());
    debug_assert_eq!(out.len(), x.len());
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    // Numerically stable: log(1 + exp(x)).
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp() // softplus(x) ≈ exp(x) for very negative x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// 1.58-bit SwiGLU FFN: `out = down(silu(gate(x)) * up(x))`.
///
/// `gate_proj` / `up_proj`: `[hidden, d_model]`; `down_proj`:
/// `[d_model, hidden]`. All three are frozen ternary projections.
pub struct BitSwiGLU {
    pub gate_proj: BitLinearWeights,
    pub up_proj: BitLinearWeights,
    pub down_proj: BitLinearWeights,
    pub hidden_dim: usize,
    pub d_model: usize,
}

impl BitSwiGLU {
    pub fn new(
        gate_proj: BitLinearWeights,
        up_proj: BitLinearWeights,
        down_proj: BitLinearWeights,
    ) -> Self {
        let d_model = gate_proj.cols;
        let hidden_dim = gate_proj.rows;
        assert_eq!(up_proj.rows, hidden_dim);
        assert_eq!(up_proj.cols, d_model);
        assert_eq!(down_proj.rows, d_model);
        assert_eq!(down_proj.cols, hidden_dim);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_dim,
            d_model,
        }
    }

    /// Toy constructor with deterministic ternary weights (tests / examples).
    pub fn new_toy(d_model: usize, hidden_dim: usize, seed: u64) -> Self {
        let scale = (1.0 / d_model as f32).sqrt() * 0.5;
        let gate = crate::delta_net::make_ternary_proj(hidden_dim, d_model, seed + 1, scale);
        let up = crate::delta_net::make_ternary_proj(hidden_dim, d_model, seed + 2, scale);
        let down = crate::delta_net::make_ternary_proj(d_model, hidden_dim, seed + 3, scale);
        Self::new(gate, up, down)
    }

    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.d_model);
        assert_eq!(out.len(), self.d_model);
        let mut gate = vec![0.0f32; self.hidden_dim];
        let mut up = vec![0.0f32; self.hidden_dim];
        self.gate_proj.forward_gemv(x, &mut gate);
        self.up_proj.forward_gemv(x, &mut up);
        for i in 0..self.hidden_dim {
            gate[i] = silu(gate[i]) * up[i];
        }
        self.down_proj.forward_gemv(&gate, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_shape_and_finite() {
        let ffn = BitSwiGLU::new_toy(32, 64, 5);
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.3).sin()).collect();
        let mut out = vec![0.0; 32];
        ffn.forward(&x, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        // Zero input -> zero output (no biases anywhere).
        let mut out0 = vec![9.0; 32];
        ffn.forward(&[0.0; 32], &mut out0);
        assert!(out0.iter().all(|&v| v == 0.0));
    }
}
