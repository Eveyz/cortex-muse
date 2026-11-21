#!/usr/bin/env python3
"""Distillation training: Qwen3-teacher data -> ternary Cortex student.

Trains the 1.6B ternary student (CortexModelTorch from
train_seed_qat.py) on the JSONL produced by distill_generate.py.
This is sequence-level SFT distillation: the student learns to
produce the teacher's responses, token by token, with loss masked
to the response only (the prompt is context, not a target).

Design notes -- read before changing the loop:
  * Encoding is byte-level (list(text.encode('utf-8'))), matching
    the Rust ByteBpeTokenizer with no merges. Special ids follow
    the Rust layout: BOS 256, EOS 257, PAD 258. Consequence: a
    Chinese character costs ~3 bytes, so --seq-len must be ~3-4x
    the character length you want to keep. The production vocab
    (32768) is mostly unused without trained BPE merges; that is
    a known limitation, not a bug in this script.
  * No autocast / no mixed precision. BitLinearQAT's STE math and
    the per-row scales must match the Rust numerics exactly; FP32
    is slower but exact. If you later add bf16, re-run the
    --self-check in train_seed_qat.py first.
  * Hippocampus interface training is OFF (hippo_train=False).
    The memory category teaches the <memory_slot> *format* as
    text; the parametric write/read interface is phase-4 joint
    training (--hippo-train in train_seed_qat.py) and is not part
    of this distillation run. The hippocampus still reads during
    training (gated, detached) -- that is how the gate learns to
    use or ignore it.
  * Padding is appended at the END of each sequence. The DeltaNet
    recurrence and the ring attention are both causal, so pad
    tokens never influence the loss positions before them.
  * Loss is plain cross-entropy over response tokens (labels -100
    elsewhere). No KL to teacher logits: the vocabularies do not
    match (see distill_generate.py header).

Memory (production config, rough): FP32 master weights 6.4 GB +
grads 6.4 GB + AdamW states 12.8 GB ~= 26 GB, plus activations
(~3 GB at batch 2 x seq 512). Fits one H100 80GB; on a 40 GB
card use --batch-size 1 --grad-accum 8.

Usage (on the GPU machine):
  python distill_train.py --data distill-data.jsonl --config production \
      --epochs 2 --out student.ctxb --ckpt-dir ckpts
  python distill_train.py --data distill-data.jsonl --config micro \
      --epochs 1 --max-examples 200        # smoke test (CPU-able, slow)
  python distill_train.py --data distill-data.jsonl --config production \
      --init-from ckpts/step-500.pt --epochs 2   # resume

Output: --out (.ctxb, loadable by cortex-cli) and --ckpt-dir/*.pt
(PyTorch state dicts for resuming). Final line prints the eval
loss and the path of the exported model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seed_qat import CortexModelTorch, ModelConfig, export_to_ctxb

BOS, EOS, PAD = 256, 257, 258
IGNORE = -100


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def encode_text(s: str) -> list[int]:
    return list(s.encode("utf-8"))


def build_example(prompt: str, response: str, seq_len: int):
    """Return (ids, labels) or None if the response alone is too long.

    ids    = [BOS] + prompt bytes + response bytes + [EOS]
    labels = [-100] * (1 + len(prompt)) + response bytes + [EOS]
    If the sequence overflows, the prompt is truncated from the LEFT
    (oldest context goes first); the response is never truncated.
    """
    p_ids = encode_text(prompt)
    r_ids = encode_text(response) + [EOS]
    if 1 + len(r_ids) > seq_len:
        return None
    keep_prompt = seq_len - 1 - len(r_ids)
    p_ids = p_ids[-keep_prompt:] if keep_prompt < len(p_ids) else p_ids
    ids = [BOS] + p_ids + r_ids
    labels = [IGNORE] * (1 + len(p_ids)) + r_ids
    return ids, labels


def load_dataset(path: Path, seq_len: int, eval_frac: float,
                 max_examples: int | None, seed: int):
    examples, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex = build_example(rec.get("prompt", ""), rec.get("response", ""), seq_len)
            if ex is None:
                skipped += 1
                continue
            examples.append(ex)
            if max_examples and len(examples) >= max_examples:
                break
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_frac)) if eval_frac > 0 else 0
    eval_set = examples[:n_eval]
    train_set = examples[n_eval:]
    print(f"loaded {len(examples)} examples "
          f"({len(train_set)} train / {len(eval_set)} eval, {skipped} skipped: too long)",
          flush=True)
    return train_set, eval_set


def collate(batch):
    max_len = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
    for i, (ex_ids, ex_labels) in enumerate(batch):
        ids[i, :len(ex_ids)] = torch.tensor(ex_ids)
        labels[i, :len(ex_labels)] = torch.tensor(ex_labels)
    return ids, labels


def batches(examples, batch_size: int, rng: random.Random):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [examples[i] for i in idx[start:start + batch_size]]


# ---------------------------------------------------------------------------
# Eval + sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, eval_set, batch_size: int, device) -> float:
    if not eval_set:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(eval_set), batch_size):
        ids, labels = collate(eval_set[start:start + batch_size])
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids[:, :-1], training=False)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
            ignore_index=IGNORE,
        )
        n_tok = (labels[:, 1:] != IGNORE).sum().item()
        total += loss.item() * n_tok
        count += n_tok
    model.train()
    return total / max(1, count)


@torch.no_grad()
def generate_sample(model, cfg, prompt: str, max_new: int, device) -> str:
    """Greedy decode (full-sequence forward each step: O(T^2), fine
    for short eval samples). Returns the decoded response text."""
    model.eval()
    ids = [BOS] + encode_text(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x, training=False)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS:
            break
        ids.append(nxt)
    model.train()
    body = bytes(b for b in ids[1:] if b < 256)
    return body.decode("utf-8", errors="replace")


SAMPLE_PROMPTS = [
    "请一步一步解答这道题，最后一个字给出答案：小明有 12 个苹果，吃了 3 个，又买了 5 个，现在有几个苹果？",
    "写出满足要求的 Python 代码：\n实现一个函数 binary_search(arr, target)，arr 为升序列表，返回 target 的下标，找不到返回 -1。",
    "请写一段关于「光合作用的过程」（生物）的教科书式讲解。",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lr_schedule(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def train(args) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    cfg = ModelConfig.micro() if args.config == "micro" else ModelConfig.production()
    if device.type == "cpu" and args.config == "production":
        print("WARNING: production config on CPU will be extremely slow; "
              "use --config micro for a smoke test.", flush=True)

    model = CortexModelTorch(cfg).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"resumed from {args.init_from}", flush=True)

    train_set, eval_set = load_dataset(
        Path(args.data), args.seq_len, args.eval_frac,
        args.max_examples, args.seed,
    )
    if not train_set:
        print("no training examples", file=sys.stderr)
        return 2

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_set) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"total optimizer steps: {total_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs}), warmup {warmup_steps}", flush=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 1)
    step = 0
    micro = 0
    t0 = time.time()
    tokens_seen = 0
    model.train()
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in batches(train_set, args.batch_size, rng):
            ids, labels = collate(batch)
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids[:, :-1], training=True)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                ignore_index=IGNORE,
            ) / args.grad_accum
            loss.backward()
            tokens_seen += int((labels[:, 1:] != IGNORE).sum().item())
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            lr = lr_schedule(step, total_steps, warmup_steps, args.lr, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {loss.item() * args.grad_accum:.4f} "
                      f"lr {lr:.2e} tok/s {tokens_seen / max(1, el):.0f}",
                      flush=True)
            if args.eval_every and step % args.eval_every == 0:
                ev = eval_loss(model, eval_set, args.batch_size, device)
                print(f"--- eval step {step}: loss {ev:.4f}", flush=True)
                for p in SAMPLE_PROMPTS:
                    print(f"[sample] {p[:40]}... -> "
                          f"{generate_sample(model, cfg, p, 64, device)[:200]}", flush=True)
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step},
                    ckpt_dir / f"step-{step}.pt",
                )
                if args.export_every:
                    export_to_ctxb(model, ckpt_dir / f"step-{step}.ctxb")
            if step >= total_steps:
                done = True
                break

    ev = eval_loss(model, eval_set, args.batch_size, device)
    print(f"final eval loss: {ev:.4f}", flush=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step},
        ckpt_dir / "final.pt",
    )
    out = export_to_ctxb(model, args.out)
    print(f"exported {out} ({out.stat().st_size} bytes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Distill Qwen3-teacher data into the ternary Cortex student")
    ap.add_argument("--data", required=True, help="JSONL from distill_generate.py")
    ap.add_argument("--config", choices=["micro", "production"], default="production")
    ap.add_argument("--out", default="student.ctxb")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument("--init-from", default=None, help="resume from a .pt checkpoint")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--export-every", action="store_true", help="also export .ctxb at each eval")
    ap.add_argument("--max-examples", type=int, default=None, help="cap dataset size (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
