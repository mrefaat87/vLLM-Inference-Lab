"""Mix workload.

Weighted blend of N child workloads. Implemented as a *merge of sorted
arrival streams* rather than independent Poisson processes — that way
the mixture's overall arrival rate matches the sum of children's rates
and the determinism guarantee transfers from the children.
"""

from __future__ import annotations

import heapq
from collections.abc import AsyncIterator
from dataclasses import dataclass

from experiments.workloads.base import Request, WorkloadGenerator


@dataclass(frozen=True)
class WeightedChild:
    workload: WorkloadGenerator
    weight: float = 1.0  # currently advisory; child rates already encode shares


class MixWorkload(WorkloadGenerator):
    name = "mix"

    def __init__(self, children: list[WeightedChild]) -> None:
        if not children:
            raise ValueError("MixWorkload requires at least one child")
        self._children = children

    async def requests(self, duration_s: float) -> AsyncIterator[Request]:
        # Pre-materialize each child stream into a list, then merge by
        # arrival_offset. This is fine for typical run lengths (seconds to
        # minutes); for very long runs a true k-way async merge would
        # use less memory but add a lot of complexity.
        child_streams: list[list[Request]] = []
        for child in self._children:
            stream: list[Request] = []
            async for r in child.workload.requests(duration_s=duration_s):
                stream.append(r)
            child_streams.append(stream)

        # k-way merge by arrival_offset_s. Pre-build heap with the head of
        # each stream; tie-break on (stream_index, request_id) for stability.
        heap: list[tuple[float, int, int, Request]] = []
        cursors = [0] * len(child_streams)
        for i, s in enumerate(child_streams):
            if s:
                heap.append((s[0].arrival_offset_s, i, 0, s[0]))
        heapq.heapify(heap)
        while heap:
            t, stream_i, idx, req = heapq.heappop(heap)
            # Re-label so callers can see which child a request came from
            # without parsing IDs.
            label = f"{self._children[stream_i].workload.name}+{req.label}"
            yield Request(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_tokens=req.prompt_tokens,
                max_new_tokens=req.max_new_tokens,
                arrival_offset_s=t,
                label=label,
                conversation_id=req.conversation_id,
                metadata={**req.metadata, "source": self._children[stream_i].workload.name},
            )
            cursors[stream_i] += 1
            nxt = cursors[stream_i]
            stream = child_streams[stream_i]
            if nxt < len(stream):
                heapq.heappush(heap, (stream[nxt].arrival_offset_s, stream_i, nxt, stream[nxt]))
