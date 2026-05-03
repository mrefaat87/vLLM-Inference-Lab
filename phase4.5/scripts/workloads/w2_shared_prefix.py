"""W2 — Shared-prefix RAG-style workload (KV routing's home turf).

A fixed ~2KB system prompt that every request begins with, followed by a
variable user message. If KV-aware routing works, the system prompt is
prefilled once per worker and then reused across all requests routed there.
With round-robin, cache hit rate degrades to 1/N. With KV-aware, it should
stay near 100% on the system prompt blocks.

Used in experiments A, B, C, E.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass


# ~2KB shared system prompt. Realistic shape: persona + tool definitions +
# behavioral guardrails — what you'd see in an agentic / RAG deployment.
_SYSTEM_PROMPT = """You are a senior staff engineer assistant integrated into an internal developer platform. You have access to the following capabilities and must adhere to the policies below.

Capabilities:
- search_codebase(query: string, repo: string) -> list of file paths and snippets matching the query.
- read_file(path: string, repo: string, start_line?: int, end_line?: int) -> file contents in the requested range.
- run_tests(repo: string, scope?: string) -> test results, including failing tests with stack traces.
- post_pr_comment(pr: int, body: string) -> creates a review comment on the specified pull request.
- query_metrics(metric: string, window?: string) -> time-series values from the internal monitoring system.

Policies:
1. Never modify production configuration without explicit approval cited in the request.
2. When proposing code changes, prefer minimal, reviewable diffs over speculative refactors.
3. Cite the exact file path and line range when referencing code. Use the search_codebase tool first; do not invent paths.
4. If a request is ambiguous, ask one clarifying question before producing output. Do not chain multiple clarifications.
5. For incidents, lead with severity assessment, then immediate mitigation, then root cause hypothesis.
6. Respect data sensitivity: never quote secrets, credentials, or PII back to the user even if visible in code.
7. When using metrics, always report the time window and aggregation (p50/p99/avg). A bare number is not actionable.
8. Refuse out-of-scope requests politely and suggest the right team or system.

Output format: Markdown. Use level-2 headers for sections, fenced code blocks with language tags for code, and inline code for symbols. Keep responses focused; verbosity is a cost, not a feature.

Now respond to the following user message:"""


_USER_MESSAGES = [
    "Why is the p99 TTFT for the inference API spiking after the last deploy? Pull the relevant metrics for the past hour.",
    "Refactor the authentication middleware in our gateway service. The current code has a deadlock under high contention.",
    "Audit the database query logs from yesterday. Anything that looks like an N+1 query?",
    "We're seeing increased 503 errors from the recommendation service. Investigate.",
    "Add observability to the new feature flag system. I want to track flag-evaluation latency.",
    "Suggest a caching strategy for the user-profile lookup. We're hitting the DB ~10k times per second.",
    "The latest migration is failing on staging. Show me the diff and the failure mode.",
    "Build a dashboard for tracking SLO adherence per service tier. Include error budget burn-down.",
    "What's the impact of switching from synchronous to async webhook delivery? Pros, cons, what could break.",
    "Diagnose the OOM in the analytics worker. Heap dump is at /var/dumps/analytics-2024-Q2.hprof.",
]


@dataclass
class Prompt:
    text: str            # full prompt: system + user
    max_tokens: int
    label: str


def generate(n: int, seed: int = 42) -> list[Prompt]:
    """Generate n shared-prefix prompts.

    Same system prompt every time (the "shared prefix"), variable user message.
    Output cap 200 tokens — slightly larger than W1 to mirror RAG response shape.
    """
    rng = random.Random(seed)
    out: list[Prompt] = []
    for _ in range(n):
        user = rng.choice(_USER_MESSAGES)
        # Add minor variability to the user message so we don't accidentally
        # cache the FULL prompt (then the experiment is trivial). The cache
        # benefit must come from the shared system prompt, not the user msg.
        suffix_words = rng.randint(10, 30)
        suffix = " ".join(["context"] * suffix_words)
        text = f"{_SYSTEM_PROMPT}\n\n{user} {suffix}"
        out.append(Prompt(text=text, max_tokens=200, label="W2"))
    return out


def _main() -> int:
    ap = argparse.ArgumentParser(description="Emit W2 prompts as JSONL.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prompts = generate(args.n, args.seed)
    if args.dry_run:
        p = prompts[0]
        print(f"label={p.label} max_tokens={p.max_tokens} char_len={len(p.text)}")
        print(f"first 200 chars: {p.text[:200]}")
        print(f"...last 200 chars: {p.text[-200:]}")
        return 0
    for p in prompts:
        print(json.dumps({"text": p.text, "max_tokens": p.max_tokens, "label": p.label}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
