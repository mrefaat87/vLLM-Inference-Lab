"""W3 — Bursty long-prompt mix (disaggregation's home turf).

80% of requests are W1-shaped (random short). 20% are 8K-input × 100-output
"long" requests dropped in as bursts every ~5s. The long requests should
saturate prefill on a co-located worker, blocking decode tokens for the
short requests on the same worker — manifesting as ITL spikes.

Disaggregated topology should be immune: prefill workers absorb the long
prompt, decode workers continue streaming the short requests' tokens.

Used in experiments A, B, C.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass

try:
    from .w1_random import generate as w1_generate, Prompt
except ImportError:
    # Allow running as a standalone script: `python3 w3_bursty.py`
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from w1_random import generate as w1_generate, Prompt  # type: ignore


def _make_long_prompt(rng: random.Random) -> Prompt:
    """8K-input × 100-output. The input is roughly 6000 random words (~8000
    tokens) — large enough that prefill takes meaningful GPU time and blocks
    decode on a co-located worker."""
    word_pool = "data record entry log line event session user request response".split()
    words = [rng.choice(word_pool) for _ in range(6000)]
    return Prompt(text=" ".join(words), max_tokens=100, label="W3-LONG")


def generate(n: int, seed: int = 42, long_ratio: float = 0.2) -> list[Prompt]:
    """Generate n prompts with `long_ratio` fraction being 8K long-prompt
    bursts. The order interleaves long and short so the load tester sees a
    realistic mix when feeding requests at a steady arrival rate."""
    rng = random.Random(seed)
    long_count = int(n * long_ratio)
    short_count = n - long_count
    short = w1_generate(short_count, seed=seed)
    longs = [_make_long_prompt(rng) for _ in range(long_count)]
    # Interleave: insert a long prompt every (n // long_count) short ones.
    # This produces "bursts" against an otherwise steady short-prompt stream.
    out: list[Prompt] = []
    short_iter = iter(short)
    if long_count == 0:
        return short
    spacing = max(1, short_count // long_count)
    short_emitted = 0
    for long_p in longs:
        for _ in range(spacing):
            try:
                out.append(next(short_iter))
                short_emitted += 1
            except StopIteration:
                break
        out.append(long_p)
    # Drain any remaining shorts
    out.extend(list(short_iter))
    return out


def _main() -> int:
    ap = argparse.ArgumentParser(description="Emit W3 prompts as JSONL.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-ratio", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prompts = generate(args.n, args.seed, args.long_ratio)
    if args.dry_run:
        labels = [p.label for p in prompts]
        print(f"total={len(prompts)} short={sum(1 for l in labels if l.startswith('W1'))} long={sum(1 for l in labels if l == 'W3-LONG')}")
        print(f"label sequence (first 20): {labels[:20]}")
        return 0
    for p in prompts:
        print(json.dumps({"text": p.text, "max_tokens": p.max_tokens, "label": p.label}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
