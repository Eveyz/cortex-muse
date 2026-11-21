//! cortex-sleep: night-time consolidation (Phase 4).
//!
//! * [`MezoScaleTuner`] — MeZO zero-order tuning of the continuous
//!   parameters `φ` (row scales + RMS weights) only. No autograd.
//! * [`ReplayBuffer`] — 80% WAL high-surprise + 20% anchor replay.
//! * [`consolidate_hippocampus`] — Gram–Schmidt row dedup + soft
//!   singular suppression for the hippocampus.

pub mod mezo;
pub mod orthogonalize;
pub mod replay;

pub use mezo::{collect_phi, scatter_phi, MezoRng, MezoScaleTuner, MezoStepReport};
pub use orthogonalize::{consolidate_hippocampus, orthogonalize_rows, soft_singular_suppress};
pub use replay::ReplayBuffer;
