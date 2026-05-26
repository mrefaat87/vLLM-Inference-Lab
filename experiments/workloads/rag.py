"""RAG / long-context QA workload.

Profile (matches production retrieval-augmented generation traffic):
  - Poisson arrivals at ``rate_rps`` (default 4/sec — RAG is heavier per
    request than chat, so fewer in flight at the same GPU budget).
  - **Large** prompts: log-normal with median 8000 tokens, sigma 0.5.
    Most prompts land in the 5–15k range with a thin tail to 20k. This is
    the user query (~50 tok) plus a retrieved context window.
  - Short outputs: log-normal, median 200 tokens, sigma 0.5. Answers are
    grounded in the retrieved context, so they're shorter than free-form
    generation.
  - **Anti-cache property:** the retrieved context is composed of N
    document chunks sampled *without replacement* from a large fixed pool.
    The composition of any two requests is unique with overwhelming
    probability, so prefix-cache hit rate should be ≈0%. This is what
    makes RAG the natural opposite of agentic-coding's long shared
    system prompt — same prompt size, very different cache behavior.

Why this matters for engine comparison: vLLM and SGLang differ
significantly on raw prefill throughput when the prefix-cache cannot
help. Roofline predictions for prefill-bound regimes are validated here.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from experiments.workloads.base import Request, WorkloadGenerator

# Size of the synthetic document pool. Big enough that without-replacement
# sampling of ~50 chunks per request collides only astronomically rarely.
_DEFAULT_POOL_SIZE = 100_000

# Approx tokens per chunk in our synthetic representation (used to convert
# a target prompt length into a chunk count).
_TOKENS_PER_CHUNK = 160


@dataclass(frozen=True)
class RagParams:
    """Default medians anchored to the Mooncake / Kimi production trace
    (arXiv:2407.00079): average input 7 590 tokens, average output
    182 tokens, ratio ~720. Our 8 000 / 200 sits squarely on those
    medians and matches the calc's "rag" preset 1:1.
    """

    rate_rps: float = 4.0
    prompt_median_tokens: float = 8000.0
    prompt_sigma: float = 0.5
    output_median_tokens: float = 200.0
    output_sigma: float = 0.5
    chunk_pool_size: int = _DEFAULT_POOL_SIZE
    tokens_per_chunk: int = _TOKENS_PER_CHUNK


class RagWorkload(WorkloadGenerator):
    name = "rag"

    def __init__(self, seed: int = 1, params: RagParams | None = None) -> None:
        self._seed = seed
        self._params = params or RagParams()
        self._rng = np.random.default_rng(seed)

    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        p = self._params
        rng = self._rng
        mu_prompt = math.log(p.prompt_median_tokens)
        mu_output = math.log(p.output_median_tokens)
        t = 0.0
        next_id = 0
        while True:
            delta = float(rng.exponential(scale=1.0 / p.rate_rps))
            t += delta
            if t >= duration_s:
                return
            next_id += 1

            prompt_tokens = max(
                p.tokens_per_chunk,
                int(math.exp(rng.normal(mu_prompt, p.prompt_sigma))),
            )
            output_tokens = max(1, int(math.exp(rng.normal(mu_output, p.output_sigma))))

            # How many chunks to retrieve to hit the target prompt size.
            n_chunks = max(1, prompt_tokens // p.tokens_per_chunk)
            # Sample chunk IDs WITHOUT replacement so the composition is unique.
            chunk_ids = rng.choice(p.chunk_pool_size, size=n_chunks, replace=False)
            prompt = _synthesize_rag_prompt(chunk_ids, prompt_tokens)

            yield Request(
                request_id=f"rag-{self._seed}-{next_id}",
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_tokens,
                arrival_offset_s=t,
                label="rag.qa",
            )


def _synthesize_rag_prompt(chunk_ids: np.ndarray, target_tokens: int) -> str:
    """Build a prompt whose leading content reflects the (unique) chunk set.

    The chunk IDs appear at the *front* of the prompt so that any two
    requests differ in their first few tokens — that's the measurable
    invariant that defeats prefix caches. A trailing question template
    is appended so the engine sees a realistic instruction shape.
    """
    head = " ".join(f"doc{int(c)}" for c in chunk_ids[: min(len(chunk_ids), 32)])
    # Fill body with the chunk IDs again to approximate the target size; we
    # don't need exact token counts — prompt_tokens in the Request is the
    # authoritative number used for analysis.
    body = " ".join(f"chunk{int(c)}_token_x" for c in chunk_ids)
    question = " Question: based on the passages above, what is the answer?"
    return f"{head} | {body}{question}"
