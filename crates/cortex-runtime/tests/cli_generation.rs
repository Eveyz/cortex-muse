//! DoD 5: build the workspace and stream-generate 32 tokens from
//! the `cortex-cli` binary, printing latency and memory statistics.

use std::process::Command;

#[test]
fn test_end_to_end_cli_generation() {
    let bin = env!("CARGO_BIN_EXE_cortex-cli");
    let out = Command::new(bin)
        .args([
            "--prompt",
            "Hello, Cortex",
            "--tokens",
            "32",
            "--non-interactive",
        ])
        .output()
        .expect("failed to run cortex-cli");
    assert!(
        out.status.success(),
        "cortex-cli exited with {}",
        out.status
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}");
    assert!(
        stdout.contains("generated 32 tokens"),
        "expected 32-token generation line, got:\n{stdout}"
    );
    assert!(
        stdout.contains("ms/token"),
        "expected latency line, got:\n{stdout}"
    );
    assert!(
        stdout.contains("state_bytes="),
        "expected memory stats, got:\n{stdout}"
    );
}
