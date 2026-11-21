//! cortex-runtime: full CortexModel assembly, Byte-BPE tokenizer,
//! day/night FSM, and the `cortex-cli` binary.

pub mod fsm;
pub mod model;
pub mod tokenizer;
pub mod tools;

pub use fsm::{FsmEvent, FsmState, RuntimeFsm};
pub use model::{sample_token, CortexConfig, CortexModel, GenStats, SamplingParams};
pub use tokenizer::{ByteBpeTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID};
pub use tools::{AgentLoop, AgentOutcome, ToolRegistry};
