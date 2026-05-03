"""W4 — Cold-start probe.

A single small prompt fired immediately after a pod becomes Ready. Used only
in experiments A (baseline numbers) and D (ModelExpress vs Phase 4.1 streamer).

The interesting metric here is NOT the inference time — it's:
  - pod_created_at  → pod_ready_at        (image pull + init containers)
  - pod_ready_at    → first_token_at      (model load + warmup)

The load tester records both. The prompt itself is intentionally trivial so
inference cost doesn't pollute the cold-start signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass


@dataclass
class Prompt:
    text: str
    max_tokens: int
    label: str


def generate() -> list[Prompt]:
    return [Prompt(text="Say hello.", max_tokens=10, label="W4-COLD")]


def _main() -> int:
    ap = argparse.ArgumentParser(description="Emit W4 cold-start probe as JSONL.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prompts = generate()
    if args.dry_run:
        p = prompts[0]
        print(f"label={p.label} max_tokens={p.max_tokens} text={p.text}")
        return 0
    for p in prompts:
        print(json.dumps({"text": p.text, "max_tokens": p.max_tokens, "label": p.label}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
