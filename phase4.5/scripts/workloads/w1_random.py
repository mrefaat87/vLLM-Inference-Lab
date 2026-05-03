"""W1 — Random short prompts (control workload).

50-500 token inputs, ~100 token outputs, no shared prefix between requests.
Used as the cache-miss control: the router can't help, disagg can't help,
spec decode can still help. Used in experiments A, B, C, E.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass


# Word pool — common English so the tokenizer behaves predictably across models.
# Drawn at random with replacement; not meant to be coherent prose. We're
# stressing the engine, not generating real content.
_WORDS = (
    "the quick brown fox jumps over lazy dog explain analyze compare contrast "
    "describe summarize recommend evaluate justify discuss outline list write "
    "code function class method variable loop condition return value type "
    "system server cluster network packet latency throughput bandwidth queue "
    "memory cache buffer thread process schedule batch token model inference "
    "prompt response stream chunk connection request retry timeout error log "
    "metric gauge histogram counter trace span span context resource limit"
).split()


@dataclass
class Prompt:
    text: str
    max_tokens: int
    label: str  # short tag for analysis (e.g., "W1-S" for short)


def generate(n: int, seed: int = 42) -> list[Prompt]:
    """Generate n random short prompts.

    ~70% short (50-150 input tokens), ~30% medium (150-500). No shared prefix.
    Output cap is uniform 100 tokens — keeps decode work comparable so TTFT
    is a clean signal vs. ITL.
    """
    rng = random.Random(seed)
    out: list[Prompt] = []
    for _ in range(n):
        if rng.random() < 0.7:
            word_count = rng.randint(40, 110)  # ~50-150 tokens (1 word ≈ 1.3 tokens)
            label = "W1-S"
        else:
            word_count = rng.randint(110, 380)  # ~150-500 tokens
            label = "W1-M"
        words = [rng.choice(_WORDS) for _ in range(word_count)]
        out.append(Prompt(text=" ".join(words), max_tokens=100, label=label))
    return out


def _main() -> int:
    ap = argparse.ArgumentParser(description="Emit W1 prompts as JSONL.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first prompt human-readable and exit 0.")
    args = ap.parse_args()
    prompts = generate(args.n, args.seed)
    if args.dry_run:
        p = prompts[0]
        print(f"label={p.label} max_tokens={p.max_tokens} char_len={len(p.text)}")
        print(f"first 200 chars: {p.text[:200]}")
        return 0
    for p in prompts:
        print(json.dumps({"text": p.text, "max_tokens": p.max_tokens, "label": p.label}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
