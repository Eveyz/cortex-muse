//! cortex-kernels: 1.58-bit ternary linear layers, ops, `.ctxb` format,
//! and Gated DeltaNet SSM (kept here for compatibility; the spec-named
//! `BitDeltaNetLayer` lives in `cortex-ssm` and wraps this layer).

pub mod bit158;
pub mod ctxb;
pub mod delta_net;
pub mod ops;

pub use bit158::{BitLinearWeights, CacheLine16x16, GemvKernel, Tile16x4};
pub use ctxb::{crc32, read_ctxb, write_ctxb, CtxbError, CtxbHeader, CtxbTensor, CtxbTensorKind};
pub use delta_net::{
    alpha_from_logit, l2_normalize_inplace, make_ternary_proj, random_ternary_matrix, rms_norm,
    sigmoid, softplus, xorshift64star, DeltaScratch, DeltaState, GatedDeltaNetConfig,
    GatedDeltaNetLayer, GatedDeltaNetStack,
};
pub use ops::{silu, BitSwiGLU};
