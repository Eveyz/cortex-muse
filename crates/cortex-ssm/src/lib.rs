//! cortex-ssm: Gated DeltaNet O(1) state layer + INT8 ring sliding-window attention.
//!
//! * [`BitDeltaNetLayer`] — layers 1–3 of a macro-block (wraps the
//!   tested `cortex_kernels::GatedDeltaNetLayer` with square heads).
//! * [`RingSlidingAttention`] — branch A of layer 4, fixed `W=512`
//!   INT8 KV ring, memory constant in conversation length.

pub mod delta;
pub mod ring;

pub use delta::{BitDeltaNetConfig, BitDeltaNetLayer};
pub use ring::{RingAttentionConfig, RingSlidingAttention};
