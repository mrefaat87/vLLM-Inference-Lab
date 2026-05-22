"""Build a narrative HTML from the inference_lab_results notebook.

We pull the chart PNGs from the notebook outputs and embed them as base64 so
the result is a single self-contained file (same shape as the Phase 3 dashboard).
The prose is intentionally hand-crafted: the notebook reads test-by-test, but
the HTML reads as a story arc through the lab.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent  # repo root (this script lives in tools/)
NB_PATH = ROOT / "lab" / "inference_lab_results.ipynb"
OUT = ROOT / "lab" / "inference_lab_results.html"


def extract_images(nb):
    """Return {cell_index: base64_png} for every code cell that produced an image."""
    images = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                png = data["image/png"]
                # Notebook may store as list-of-lines or a single string.
                if isinstance(png, list):
                    png = "".join(png)
                images[i] = png
                break
    return images


def img_tag(images, idx, alt):
    """Inline a chart as a base64 data URI."""
    if idx not in images:
        return f"<!-- missing image for cell {idx} -->"
    return (
        f'<figure class="chart"><img src="data:image/png;base64,{images[idx]}" '
        f'alt="{alt}"/></figure>'
    )


def build_derived_tbt_chart():
    """Render a TBT-vs-RPS chart from test 25 / 29A / 30A and return base64 PNG.

    TBT is not directly instrumented in stage 2 — we derive it from
    (total_p50 - ttft_p50) / avg_output_tokens. The chart compares T4 FP16,
    T4 FP8 E5M2, and A10G FP8 E4M3 across the open-loop RPS ramp.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [
        ("T4 FP16 (baseline)",       "stage2_test25_results.json",  "#58a6ff"),
        ("T4 FP8 E5M2",              "stage2_test29a_results.json", "#f85149"),
        ("A10G FP8 E4M3",            "stage2_test30a_results.json", "#3fb950"),
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#161b22")
    ax.set_facecolor("#0d1117")

    for label, path, color in configs:
        rows = json.loads((ROOT / "stage2" / path).read_text())
        xs, ys = [], []
        for r in rows:
            if r.get("successes", 0) == 0:
                continue
            ttft = r.get("ttft_p50", 0)
            tot = r.get("total_p50", 0)
            decode = tot - ttft
            # Output tokens generated per request, averaged from system throughput.
            sys_tps = r.get("system_tok_s", 0)
            wall = r.get("wall_duration_sec", 60)
            n_req = max(r.get("total_requests", 1), 1)
            out_tok = (sys_tps * wall) / n_req
            if out_tok <= 1 or decode <= 0:
                continue
            tbt_ms = (decode / (out_tok - 1)) * 1000
            xs.append(r["rps"])
            ys.append(tbt_ms)
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7,
                color=color, label=label)

    ax.set_xlabel("Sustained RPS (open-loop)", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Derived TBT (ms / output token)", color="#c9d1d9", fontsize=11)
    ax.set_title(
        "Derived time-between-tokens vs RPS — KV-cache dtype impact on decode speed",
        color="#e6edf3", fontsize=12, pad=12,
    )
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#21262d", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#c9d1d9", fontsize=10, loc="upper left")
    # Annotate the surprising T4 regression at 12 RPS.
    ax.annotate(
        "T4 FP8 regresses past 8 RPS\n(cast-back overhead dominates)",
        xy=(12, 109.5), xytext=(7.2, 150),
        color="#f85149", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
    )
    ax.annotate(
        "A10G stays flat ~12ms across the ramp",
        xy=(12, 11.8), xytext=(13, 35),
        color="#3fb950", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.2),
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#161b22")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(images):
    # Narrative is structured as Acts. Each Act tells one piece of the story
    # and pulls in 1-3 charts. We avoid the "Test N: scenario / chart / insights"
    # template — instead we embed the question, then the picture, then the lesson.
    derived_tbt_b64 = build_derived_tbt_chart()
    derived_tbt_tag = (
        f'<figure class="chart"><img src="data:image/png;base64,{derived_tbt_b64}" '
        f'alt="Derived TBT vs RPS across T4 FP16, T4 FP8 E5M2, A10G FP8 E4M3"/></figure>'
    )

    parts = []
    parts.append(HEAD)
    parts.append(HERO)
    parts.append('<div class="content">')
    parts.append(TOC)

    # ============================================================
    parts.append("""
<h2 id="sec-1"><span class="section-num">1</span> The question that started everything</h2>
<p>Stage 1 was a laptop, an M4, and Ollama. Stage 2 was a T4 on AWS Spot. The point of the
lab wasn't to ship a service &mdash; it was to <em>see</em> the things production inference systems
are designed around: continuous batching, paged KV cache, prefix reuse, the prefill/decode
asymmetry, the cliff between healthy and dead.</p>

<p>The first experiment was the simplest possible one: send three concurrent requests to a server
that doesn't batch. What happens?</p>

""" + img_tag(images, 3, "Ollama waterfall: three sequential requests on Apple M4") + """

<div class="callout callout-info">
<strong>The staircase is the point.</strong> Each request only starts after the previous one fully
finishes &mdash; prefill <em>and</em> every decode token. The gap between first-tokens (~81s) equals
a full request duration. Request 2 waits 177s for its first token, not because the model is slow
but because it's third in line. This is a single-instance target group with no ASG. Throughput
scales by <em>adding workers</em>, not by being clever with one.
</div>

<p>vLLM, on the same prompt, looks like a different machine entirely:</p>

""" + img_tag(images, 6, "vLLM continuous batching: 5 concurrent requests on T4") + """

<p>All five first-tokens arrive within milliseconds of each other. The gap-to-total ratio is
near zero. This is continuous batching &mdash; the city bus that picks up passengers at every stop
instead of waiting for everyone to board. Every chart that follows assumes this primitive works;
it's the foundation under everything else.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-2"><span class="section-num">2</span> How far can one GPU actually go?</h2>

<p>If batching is "free", maybe we should just keep stuffing requests in. The next test ramped
identical short-prompt requests from N=1 to N=80 to see where the GPU stops thanking us.</p>

""" + img_tag(images, 9, "Throughput vs concurrency, short prompts up to N=80") + """

<p>System throughput peaked at <strong>N=60 (1,132 tok/s)</strong> and then started <em>declining</em>.
Per-request tok/s halved by N=40. Zero failures &mdash; KV cache only hit 1.7% because the responses
were 30 tokens each. The cliff isn't memory; it's scheduler overhead and shared bandwidth.</p>

<div class="callout callout-info">
<strong>Auto Scaling parallel:</strong> this is exactly why you don't set ASG max to infinity.
There's a coordination tax. The right fleet size is the one that maximizes useful throughput,
not the one that maximizes instance count.
</div>

<p>Short prompts mask the real shape of the curve. Repeating the experiment with realistic
~200-in / ~200-out workloads exposes the <em>knee</em>:</p>

""" + img_tag(images, 15, "Throughput knee with medium prompts: efficiency drops past N=8-16") + """

<p>System throughput climbs 25&times; (44 &rarr; 1,105 tok/s) but efficiency collapses from 87% at
N=8 to 37% at N=64. The "knee" lives between N=8 and N=16 &mdash; that's where each new request
starts costing more than it adds. This is the number a target-tracking scaling policy should
fire on, not raw RPS.</p>

<p>And once we mix prompt sizes the way real traffic does, vLLM does something subtle and
elegant on its own:</p>

""" + img_tag(images, 12, "Mixed-prompt scheduling: short prompts get first-token priority emergently") + """

<div class="callout callout-success">
<strong>Emergent priority, not configured priority.</strong> Short prompts barely degrade
(+20% TTFT). Medium and long prompts degrade ~85%. Nobody told vLLM to prioritize anyone &mdash;
chunked prefill just finishes faster for shorter inputs, so they jump the line. The scheduler
acts like SJF (shortest-job-first) without being asked to.
</div>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-3"><span class="section-num">3</span> Quantization: shrinking the model without shrinking what it knows</h2>

<p>The T4 has 16GB of VRAM. A FP16 7B model needs ~14GB just for weights, leaving almost
nothing for KV cache. Quantization is how you get serving headroom on small GPUs &mdash;
the question is what you give up for it.</p>

""" + img_tag(images, 18, "Quantization speed: AWQ, AWQ_Marlin, GPTQ Int8") + """

<p>Same 5-concurrent load, three weight formats. The Marlin kernel reads the same AWQ weights
but runs prefill <strong>47% faster</strong> (TTFT 0.29s vs 0.55s) &mdash; same model, better compute.
Int8 is ~40% slower on decode and uses 2&times; the VRAM, halving the KV cache room. All three batch
identically because batching is engine-level, not format-level.</p>

<p>So 4-bit is faster, smaller, and gives more concurrency room. The obvious follow-up: does
it still get the right answer?</p>

""" + img_tag(images, 21, "Quality benchmark: INT4 vs INT8 across 7 categories, 60 questions") + """

<div class="callout callout-success">
<strong>60/60 vs 60/60.</strong> AWQ INT4 and GPTQ INT8 scored identically across math, factual
recall, instruction following, code, MMLU-style, GSM8K-style, and HumanEval-style prompts.
INT4 was ~60% faster on every category. At 7B+ scale with modern quantization (AWQ/GPTQ),
"4-bit ruins quality" is folk wisdom from 1-3B-era results. Pick INT4 for the speed and the
KV headroom; the quality cost is noise.
</div>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-4"><span class="section-num">4</span> The KV cache is a CDN</h2>

<p>If the same 500-token system prompt heads every request &mdash; RAG context, chat history,
a function-calling preamble &mdash; vLLM only computes its KV once and reuses it. This is
prefix caching, and it's the closest analog to a CDN that exists inside an inference engine.</p>

""" + img_tag(images, 24, "Prefix caching ON: warm TTFT 0.19s vs cold 0.34s") + """

<p>Cold TTFT 0.34s &rarr; warm TTFT 0.19s. <strong>1.75&times; speedup</strong> on a 500-token shared
prefix; the speedup grows with prefix length. 10 concurrent requests all hit the warm cache
with a tight P50/P99 distribution. The longer the shared prompt, the more dramatic this gets &mdash;
RAG and multi-turn chat both lean on it heavily.</p>

<p>That same caching is also what lets us push much further on KV pressure than first-principles
math would suggest:</p>

""" + img_tag(images, 27, "KV cliff test: 70 concurrent long-context requests, no preemption") + """

<p>Theoretical max at 2,000 tokens/request was N=64. We pushed to N=70 long-context (~500 in,
~1,500 out) requests and saw zero preemption &mdash; the 93% prefix hit rate dropped effective
per-request KV from 2,000 to 1,500 tokens. Throughput peaked at N=60 again (the same knee from
the short-prompt run). The scheduler doesn't know whether it's serving short or long prompts;
it finds the same optimal batch shape regardless.</p>

<p>The same primitive carries multi-turn chat for free:</p>

""" + img_tag(images, 45, "Multi-turn TTFT stays flat as context grows 4.7x") + """

<p>Five conversations, four to five turns each, context growing from 182 to 854 tokens. TTFT
stayed flat at ~0.19s. Sticky sessions in an ALB are the analogy here &mdash; the same instance
keeps the user's warm state, every subsequent turn pays only for the new tokens.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-5"><span class="section-num">5</span> The wall, and what's on the other side of it</h2>

<p>To find the actual KV ceiling we had to fight the engine: GPTQ Int8 (less KV room),
prefix caching <em>off</em>, 40 unique prompts so nothing could be shared, max_tokens=1500.
This was the test that finally broke vLLM.</p>

""" + img_tag(images, 30, "KV cache hits 100%: vLLM queues, then the server crashes past N=45") + """

<div class="callout callout-warning">
<strong>vLLM chose queuing over preemption.</strong> KV pegged at 100%, <code>Waiting</code>
went 0&rarr;5, running requests kept their pages, new ones waited. No preemption logged &mdash;
evicting an in-flight decode (throwing away minutes of work) is more wasteful than delaying
a new request by seconds. N=38 succeeded, N=40 squeaked through, N=45 crashed the server outright.
The failure mode past the limit is <em>catastrophic, not graceful</em> &mdash; this is a recurring theme.
</div>

<p>Holding the line at the wall is one problem. Why does crossing it cost so much in the first
place? The next two charts explain it.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-6"><span class="section-num">6</span> Prefill and decode are not the same thing</h2>

<p>Two sub-tests on a single request: vary input from 10&rarr;1,000 tokens with output fixed,
then vary output from 10&rarr;1,000 with input fixed. The result reframes how you think about
inference cost.</p>

""" + img_tag(images, 33, "Prefill scales with input, decode scales with output, independently") + """

<p>100&times; the input grew TTFT by only 1.0&times;. Decode time was unchanged. Then 17&times; the output
grew decode time 18.7&times; while TTFT stayed flat at 0.21s. Input and output don't share a
bottleneck &mdash; prefill is GPU-compute parallel, decode is memory-bandwidth sequential.</p>

<p>That's the <em>shape</em> of the asymmetry. Here's the price tag:</p>

""" + img_tag(images, 36, "Output tokens cost ~17x more GPU time than input tokens") + """

<div class="callout callout-info">
<strong>Why API providers charge 3-5&times; more for output tokens:</strong> on this T4, output
tokens cost <strong>16.9&times;</strong> more GPU time than input (21.6 ms vs 1.3 ms per token).
The same 1,000 tokens take 2.6s heavy-input vs 15.6s heavy-output &mdash; a 6&times; wall-clock
difference. Input cost <em>decreases</em> at scale (parallelism amortizes overhead);
output cost stays constant (each token reads the full model weights). The GPU is idle ~95% of
the time during decode, waiting on memory.
</div>

<p>This is the asymmetry that justifies disaggregated inference (prefill on big GPUs, decode
on cheap memory-bandwidth-rich ones) and speculative decoding (filling decode's idle compute
with verified guesses). Both are on the Phase 5+ roadmap for exactly this reason.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-7"><span class="section-num">7</span> Two speed-neutral knobs &mdash; one is decisive anyway</h2>

<p>Sections 1&ndash;6 catalogued the things that <em>do</em> move the needle: batching, quantization,
KV cache, prefix reuse, the prefill/decode split. Before moving on to production load tests,
two more knobs needed to be ruled in or out: <strong>guided JSON output</strong> (does forcing the
model to emit a schema slow it down?) and <strong>sampling temperature</strong> (does decoding
stochastically cost more than greedy?).</p>

<p>Both turned out to be speed-neutral. But that's where the symmetry ends &mdash; one is the most
impactful production knob in this entire lab, and the other is a non-decision.</p>

<h3>Guided output: zero speed cost, 100% &rarr; 0% parse-failure rate</h3>

<p class="round-meta">Note on terminology: "failure" here is a <em>correctness</em> failure &mdash; the
server returned a 200 OK with a token stream, but the stream isn't parseable as the requested
JSON. That's different from the server-level failures discussed in Section 8 (HTTP 500, timeout,
connection refused). Both are problems; they live at different layers.</p>

""" + img_tag(images, 39, "Guided JSON output: 100% schema compliance, zero speed cost") + """

<p>Three modes, same prompt, same model:</p>
<ul>
  <li><strong>Unguided</strong> (just ask nicely in the prompt): <strong>0 of 20</strong> responses
  parsed as valid JSON. The model wraps the JSON in markdown fences, prepends "Here's the JSON:",
  adds trailing commentary, or invents fields the prompt didn't ask for.</li>
  <li><strong>JSON mode</strong> (<code>response_format=json_object</code>): <strong>20 of 20</strong>
  valid &mdash; vLLM constrains decoding so only tokens that keep the output as valid JSON get sampled.</li>
  <li><strong>Strict schema</strong> (full JSON schema with required fields): <strong>20 of 20</strong>
  valid <em>and</em> matching the schema's required-field set.</li>
</ul>

<p>All three modes ran at <strong>~45.9 tok/s</strong>. The constrained-decoding mask costs
essentially nothing because the heavy work (the forward pass through the model) is identical &mdash;
the constraint just zeros out forbidden tokens before sampling.</p>

<div class="callout callout-success">
<strong>Why this is the most impactful knob in the lab.</strong> Without guided output, you'd
build a retry loop: parse, fail, re-prompt, retry, eventually give up. At 0/20 success that
loop dominates your latency budget and your error rate. With guided output the failure mode
disappears entirely &mdash; downstream services can <em>trust</em> the response shape. This is
the inference equivalent of API Gateway request validation: enforce the contract at the edge
so nothing downstream has to handle malformed payloads. <strong>Free at runtime, transformative
at the system level.</strong>
</div>

<h3>Sampling temperature: a non-decision for speed</h3>

""" + img_tag(images, 42, "Sampling parameters: 1.3% spread across six configurations") + """

<p>Six configurations &mdash; greedy (temp=0), temp=0.3, 0.7, 1.0, 1.5, top_k=50 &mdash; ten serial
requests each. The spread across all six was <strong>1.3%</strong> (46.1 to 46.7 tok/s). TBT
varied by 0.3ms (21.53&ndash;21.82ms). Greedy was marginally faster, almost certainly because the
greedy code path skips the categorical sampling step and just calls <code>argmax</code>.</p>

<div class="callout callout-info">
<strong>Why it doesn't move:</strong> sampling is a tiny softmax + categorical draw over the
vocabulary distribution at the end of each decode step. The decode step itself reads the entire
model's weights through GPU memory &mdash; that's what dominates wall time (~95% of decode is
memory-bandwidth-bound, see Section 6). The sampling kernel is rounding error against that.
<strong>Pick temperature for the output you want, not the speed you want.</strong> If creative
diversity matters, run at temp=0.7&ndash;1.0; if reproducibility matters, run greedy. Either way,
the throughput conversation is unaffected.
</div>

<h3>The asymmetry between these two knobs</h3>

<p>Both are speed-neutral. But guided output shifts <strong>correctness from 0% to 100%</strong> &mdash;
it's a system-level capability, not a tuning parameter. Sampling temperature shifts the
<em>character</em> of the output (deterministic vs. creative) without affecting whether
downstream code can use it. In capacity-planning terms: guided output reshapes the failure
budget. Sampling temperature is a product decision that happens to be free.</p>

<p>The bigger lesson for the production stack: <em>not every knob with a flag is a tuning
opportunity</em>. Some are correctness gates with no cost (guided output, prefix caching,
PagedAttention). Some are zero-cost product choices (sampling). The actual scaling levers
&mdash; concurrency, KV-cache budget, quantization format, hardware tier &mdash; are scarcer than
the surface area of the CLI suggests.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-8"><span class="section-num">8</span> Where the production cliff actually is</h2>

<div class="callout callout-info">
<strong>Definitions for this section.</strong> A request <strong>succeeds</strong> if the server
returned a complete response &mdash; full token stream, no transport error. A request
<strong>fails</strong> if the server returned a transport-level error: HTTP 500, connection
refused, request timeout (typically &gt;30s with no response), or zero tokens streamed.
<em>Failure here means the server couldn't serve the request, not that the response was slow.</em>
SLA compliance is a separate, finer measure &mdash; tracked as the percentage of <em>successful</em>
requests that landed under a latency threshold (e.g., "96.6% under 1s TTFT" in Test 28).
A 25-second response that completes still counts as a success even though it would breach
most SLAs. The progression in this section goes: all-success-with-good-latency &rarr;
all-success-with-degraded-tails &rarr; mixed success/failure &rarr; total server crash.
</div>

<p>Open-loop load at fixed RPS, sustained 60 seconds each, ramping 1&rarr;2&rarr;4&rarr;8&rarr;12&rarr;16&rarr;20.
This is the test that tells you what number to put in the SLA.</p>

""" + img_tag(images, 48, "Tail latency fan chart: P50/P90/P95/P99 across RPS levels") + """

<div class="callout callout-warning">
<strong>The transition from "fine" to "dead" spans 4 RPS.</strong> 1&ndash;12 RPS is rock solid:
P99/P50 stays at 1.1&ndash;1.3&times;, every request succeeds. <strong>16 RPS is the cliff:</strong>
25% failures (241 of 960), P99/P50 jumps to 2.0&times;, TTFT P99 hits 0.88s. <strong>20 RPS:</strong>
0% success, server crash. SLA-safe capacity is 12 RPS for short-prompt traffic. The alarm
needs to fire well before that.
</div>

<p>To rule out a slow-burn memory leak we held 4 RPS for 15 minutes:</p>

""" + img_tag(images, 51, "Soak test: 3,600 requests, zero failures, flat metrics") + """

<p>3,600/3,600 requests, zero failures, GPU memory locked at 14,527 MiB, TTFT P50 stable at
0.225s, throughput flat at 430 tok/s. PagedAttention's memory management is leak-free. You
can park a vLLM instance at 4 RPS indefinitely.</p>

<p>The much harder question: what happens if you <em>don't</em> stop at 4 RPS?</p>

""" + img_tag(images, 54, "Graceful degradation test: vLLM crashes catastrophically past the cliff") + """

<div class="callout callout-warning">
<strong>vLLM has no built-in backpressure.</strong> 4 RPS: 100% success. 8 RPS: 100%. 14 RPS:
54% success and a mix of 500s and timeouts. 20 RPS: 0%, server dead. The 4 RPS recovery stage
found a corpse. The error progression &mdash; healthy &rarr; HTTP 500 &rarr; connection refused &rarr;
timeout &mdash; is the textbook cascading-failure pattern. <em>The server will not save itself.</em>
External infrastructure has to: rate limiting at the gateway, queue-depth alarms that scale
out before saturation, circuit breakers, fast Karpenter provisioning. This is the entire
motivation for Phases 3 and 7.
</div>

<div class="callout callout-info">
<strong>Reading note &mdash; why does the 4 RPS bar look bad?</strong> The first stage shows
TTFT P99 of 8.58s and total P99 of 13.4s, even though every one of its 240 requests
succeeded and the median TTFT was a clean 0.23s. That's <em>not</em> 4 RPS being hard &mdash;
stage 2 ran at <strong>2&times; the load</strong> with TTFT P99 of 0.28s. The first stage paid the
warmup tax: <strong>CUDA graphs compile lazily for batch sizes the server hasn't seen yet</strong>
(the same effect that produced the 9.1s TTFT spike at N=10 in Section 2), and the prefix cache
was cold. A handful of warmup outliers dragged the P99 of 240 samples. The ~37&times; P99/P50 ratio
on stage 1 vs ~1.1&times; on stage 2 is the signature. Production load tests typically discard a
30-second warmup window for exactly this reason &mdash; the story of this chart is stages 3&ndash;5
(cliff, crash, no recovery), not stage 1.
</div>

<p>Finally, the realistic mix: 40% short, 30% medium, 15% long, 10% guided JSON, 5% multi-turn,
Poisson arrivals at 6 RPS average with bursts to 12.</p>

""" + img_tag(images, 57, "Production simulation: 6 RPS sustained + 12 RPS bursts, mixed prompts") + """

<p>Mixed traffic safe capacity is <strong>4&ndash;5 RPS</strong>, dramatically lower than the
12 RPS short-prompt limit. The first burst (180&ndash;210s) at 12 RPS handled cleanly &mdash; 353 OK,
0 failures &mdash; but the cumulative effect of sustained 6 RPS plus burst aftermath overwhelmed
the server by minute 4&ndash;5. Long queries (P99 32s) dominate total time and short queries get
stuck behind them in the queue. SLA: 96.6% under 1s TTFT, 70.1% under 10s total &mdash; before
the crash.</p>

<div class="callout callout-info">
<strong>Capacity-planning lesson:</strong> short-prompt benchmarks lie. Plan with mixed traffic,
plan for bursts, plan for &lt;30s scale-out. For 100 RPS production with this stack you'd need
~20&ndash;25 T4s &mdash; or a much smaller fleet of A10G/H100s. Which leads directly to the next experiment.
</div>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-9"><span class="section-num">9</span> FP8 KV cache: a hardware-gated optimization, learned the hard way</h2>

<p>The KV cache is the master constraint of this entire lab &mdash; every previous section
ended up bumping into it from a different angle. So the obvious next move was to ask:
<strong>can we just make each KV entry smaller?</strong> vLLM exposes a single flag for this:
<code>--kv-cache-dtype fp8_e5m2</code>. The promise is seductive: each KV entry stored in 8 bits
instead of 16, doubling capacity, supposedly with no other change. The server logs even
confirm the capacity gain immediately on startup &mdash;
<code>GPU KV cache size: 255,696 tokens</code> on T4, exactly 2&times; the FP16 baseline of
127,840 tokens.</p>

<p>The first question to settle was whether this capacity is <em>real</em> &mdash; whether
the engine can actually serve traffic into the extended KV space, or whether something else
breaks first.</p>

<h3>Latency under load (T4, FP8 E5M2 vs FP16)</h3>

""" + img_tag(images, 60, "FP8 E5M2 vs FP16 latency on T4 across RPS") + """

<p>This is a replay of the Section 8 tail-latency test (open-loop RPS ramp 1&rarr;20, sustained
60s each) with the single flag flipped. The chart shows TTFT improving under FP8 at low RPS
and both formats hitting a cliff at 20 RPS. <strong>But TTFT is the wrong metric to focus on
here.</strong></p>

<div class="callout callout-warning">
<strong>Why TBT is the right metric for a KV-cache change.</strong> KV cache is read on
<em>every decode step</em> &mdash; once per output token. Prefill (TTFT) is mostly compute over
the input prompt; the KV writes during prefill are secondary. So a smaller KV dtype should
move <strong>time-between-tokens</strong>, the metric users actually feel during streaming,
much more than it moves TTFT. We didn't instrument TBT explicitly in this stage &mdash; the
test scripts captured TTFT and total time, but not per-token timestamps. Deriving TBT from
<code>(total_p50 &minus; ttft_p50) / output_tokens</code> gets us close enough to see the shape:
</div>

<table>
  <thead>
    <tr><th>Config</th><th>RPS</th><th>TTFT P50</th><th>Decode (s)</th><th>~TBT (ms)</th></tr>
  </thead>
  <tbody>
    <tr><td>T4 FP16 baseline</td><td>1</td><td>0.21s</td><td>1.77</td><td><strong>22.1</strong></td></tr>
    <tr><td>T4 FP16 baseline</td><td>8</td><td>0.24s</td><td>2.66</td><td><strong>33.7</strong></td></tr>
    <tr><td>T4 FP16 baseline</td><td>12</td><td>0.27s</td><td>3.45</td><td><strong>43.7</strong></td></tr>
    <tr><td>T4 FP8 E5M2</td><td>1</td><td>0.07s</td><td>2.32</td><td><strong>24.2</strong></td></tr>
    <tr><td>T4 FP8 E5M2</td><td>8</td><td>0.10s</td><td>3.63</td><td><strong>39.3</strong></td></tr>
    <tr><td>T4 FP8 E5M2</td><td>12</td><td>0.25s</td><td>9.59</td><td><strong>109.5</strong></td></tr>
    <tr><td>A10G FP8 E4M3</td><td>1</td><td>0.03s</td><td>0.89</td><td><strong>11.3</strong></td></tr>
    <tr><td>A10G FP8 E4M3</td><td>8</td><td>0.04s</td><td>1.07</td><td><strong>13.9</strong></td></tr>
    <tr><td>A10G FP8 E4M3</td><td>16</td><td>0.04s</td><td>1.03</td><td><strong>13.0</strong></td></tr>
  </tbody>
</table>

""" + derived_tbt_tag + """

<p class="round-meta">Chart derived from Tests 25, 29A, and 30A &mdash; the same open-loop RPS
ramp re-run on each config. Each point is
<code>(total_p50 &minus; ttft_p50) / avg_output_tokens</code>, then converted to
ms/token. Lines stop where the server started returning failures (no successful
samples to compute medians from).</p>

<div class="callout callout-warning">
<strong>The surprising T4 result: FP8 made TBT <em>worse</em>.</strong> At 1 RPS: 22.1ms FP16
vs 24.2ms FP8. At 8 RPS: 33.7 vs 39.3. By 12 RPS the FP8 server is at 110 ms/token vs FP16's
44 ms. Theory says smaller KV entries &rarr; less memory traffic &rarr; faster decode. So why did
it move the wrong way? <strong>T4 has no native FP8 compute.</strong> Every attention step
casts FP8 &rarr; FP16 in software for the matmul. The memory-traffic gain from halved entries
is real, but the cast overhead eats it and then some. On Turing, FP8 KV cache is a regression
on the metric that matters most for streaming UX.
</div>

<p>The TTFT improvement that <em>does</em> show up on T4 (0.21s &rarr; 0.07s at 1 RPS) is real
but not where the value should be coming from. Some of it is faster KV writes during prefill,
some is likely test-setup variance &mdash; the FP8 run generated ~20% more output tokens per
request than the FP16 run, suggesting the prompts and warmup states weren't perfectly matched.
Either way, TTFT is not the axis where you should expect a KV-cache optimization to pay off.</p>

<p>Where the KV-cache benefit <em>does</em> show up is on the A10G row: <strong>~11&ndash;14 ms
TBT, roughly flat across the entire RPS ramp</strong>. That's half the T4 FP16 baseline at 1
RPS and a third by 12 RPS. But it's a mixed-cause result &mdash; A10G has native FP8 tensor
cores (no cast overhead), 2&times; the memory bandwidth (600 GB/s vs 320 GB/s), and 4&times; the
KV capacity. We don't have an A10G FP16 baseline in this stage to cleanly isolate &ldquo;FP8's
contribution&rdquo; from &ldquo;Ampere's contribution.&rdquo; That isolation is on the Phase 5+
agenda, along with proper per-token TBT instrumentation.</p>

<p>One useful negative result <em>does</em> survive: <strong>both T4 formats die at exactly
20 RPS</strong>. The KV-cache size didn't move the cliff, which means at extreme load the
bottleneck stops being KV memory and becomes compute or scheduler throughput. Doubling the
cache doesn't help when the GPU is already saturated on the forward pass &mdash; the Section 8
&ldquo;sharp 12&rarr;16 RPS cliff&rdquo; is structural, not a memory problem you can optimize
around with a smaller dtype.</p>

<h3>Throughput under concurrency (T4, extending to N=128)</h3>

""" + img_tag(images, 61, "FP8 throughput on T4 scales to N=128 with zero failures") + """

<p>This one extends the Section 2 concurrency curve past where FP16 could go. FP16 crashed
beyond N&approx;64 (the KV cliff from Section 5). FP8 handled 128 concurrent requests with
<strong>zero server-level failures</strong> (using the Section 8 definition). Peak system
throughput hit 1,072 tok/s &mdash; comparable to the FP16 peak rather than double it &mdash; and the
extended range from N=80 to N=128 sustained ~950&ndash;1,040 tok/s.</p>

<p>So the 2&times; capacity is genuine on the wire. We can fit twice as many simultaneous
requests, the engine doesn't crash, and the throughput curve extends cleanly into territory
FP16 can't reach. On a hardware tier where this works, that's a real production win &mdash;
twice the headroom for traffic spikes from the same physical GPU.</p>

<h3>The third axis: accuracy</h3>

<p>Latency and throughput were the easy axes. The third one is the one that actually matters:
<strong>does the model still give correct answers?</strong> We re-ran the Section 3 quality
benchmark (60 questions, 7 categories, temperature=0) under FP8.</p>

""" + img_tag(images, 62, "FP8 E5M2 accuracy collapse on T4: 48% overall, gibberish on long outputs") + """

<p>Overall accuracy fell from ~92% to <strong>48%</strong>. But the headline number hides the
real shape of the failure. Look at category-by-category:</p>

<ul>
  <li><strong>MMLU-style (~50 token answers):</strong> 100%, fine. Short outputs survive.</li>
  <li><strong>Factual recall, basic math:</strong> partially intact &mdash; usually right, sometimes
  off by a small numerical error.</li>
  <li><strong>GSM8K-style (~500-token chain of thought):</strong> 0&ndash;10%. The model starts a
  reasoning chain, drifts, and ends with garbage.</li>
  <li><strong>HumanEval-style code generation:</strong> 0&ndash;10%. Code is well-formed for ~50
  tokens, then degrades into syntactically invalid output.</li>
</ul>

<div class="callout callout-warning">
<strong>The mantissa problem.</strong> E5M2 has 5 exponent bits and only <strong>2 mantissa
bits</strong>. That's huge dynamic range (good for activations that span many orders of
magnitude) but extremely coarse precision (bad for anything where the exact value matters).
Attention scores are softmax-normalized &mdash; they live in a narrow numerical range and care a
lot about precision. Each decode step reads the corrupted KV, computes attention over slightly
wrong values, and the error compounds across the autoregressive sequence. After ~50 tokens the
drift is small. After 500 tokens it's catastrophic. <strong>E5M2 is the wrong format for KV
cache</strong> &mdash; it optimizes for the wrong axis.
</div>

<p>The relevant alternative is <strong>E4M3</strong> (4 exponent bits, 3 mantissa bits) &mdash;
the extra mantissa bit doubles the precision of stored values at the cost of dynamic range,
which is the right tradeoff for attention. But T4 (Turing, sm_75) doesn't support E4M3 at all.
The format is hardware-gated to Ampere and newer (sm_80+). To test it, we had to leave Turing
behind.</p>

<h3>Crossing to Ampere: A10G with E4M3</h3>

<p>We rented a <strong>g5.xlarge</strong> ($0.45/hr Spot, NVIDIA A10G 24GB, Ampere sm_86, native
FP8 tensor cores) and re-ran all three sub-experiments with <code>--kv-cache-dtype fp8_e4m3</code>.
The hardware difference is meaningful in three ways simultaneously: more VRAM (24GB vs 16GB),
nearly 2&times; the memory bandwidth (600 GB/s vs 320 GB/s), and hardware-native FP8 tensor cores
that eliminate the cast-back-to-FP16-for-compute overhead the T4 paid.</p>

<p>The KV capacity expanded accordingly: <strong>508,208 tokens, 4&times; the T4 FP16 baseline.</strong>
Server logs confirmed it on startup. The question was whether the precision recovery from E4M3
was enough to fix the accuracy collapse.</p>

""" + img_tag(images, 65, "FP8 E4M3 on A10G accuracy: improved to 60% but still degraded") + """

<p>Quality recovered <em>partially</em>. Overall accuracy moved from 48% to <strong>60%</strong>.
Instruction Following jumped 40&rarr;80%. HumanEval went 0&rarr;40%. But GSM8K stayed pinned at
10%, and the long-chain failure mode persisted. <strong>The extra mantissa bit helped, but
didn't close the gap to FP16.</strong></p>

<p>That's an important result: it means the problem isn't <em>just</em> E5M2. Even with E4M3 on
hardware that supports it natively, raw FP8 KV cache without per-channel calibrated scaling
factors degrades quality. The vLLM warning at server startup &mdash; "may cause accuracy drop
without a proper scaling factor" &mdash; is exactly right. The missing piece isn't the format,
it's <strong>calibration</strong>: an offline pass that computes per-channel scaling factors
on a small calibration dataset and loads them at serve time. This is the same mechanism weight
quantization (AWQ, GPTQ) has used since the beginning &mdash; and the reason weight quantization
"just works" while KV-cache quantization doesn't.</p>

<h3>Throughput on Ampere: a different league</h3>

""" + img_tag(images, 66, "A10G throughput: 3,049 tok/s peak, 28 RPS with zero failures") + """

<p>Quality aside, the A10G's raw performance is in a different league from the T4. Peak system
throughput hit <strong>3,049 tok/s</strong> &mdash; <strong>2.8&times;</strong> the T4 E5M2 peak
and <strong>3.4&times;</strong> the T4 FP16 peak. The latency test ran from 1 to 28 RPS with
<strong>zero failures at every level</strong>, and TTFT P99 stayed under 70ms throughout.
The Section 8 cliff at 20 RPS on T4 is invisible here; the A10G simply has more headroom on
every axis.</p>

<p>Three things stack to produce this. First, native FP8 tensor cores eliminate the FP8&rarr;FP16
cast that the T4 had to do for every attention computation. Second, the 600 GB/s memory
bandwidth nearly doubles what decode (which is memory-bandwidth-bound, see Section 6) can do
per unit time. Third, the 4&times; KV capacity (508K vs 128K tokens) means the engine is never
memory-constrained at these load levels &mdash; <em>the bottleneck has shifted entirely from
memory to compute.</em></p>

<div class="callout callout-success">
<strong>The cost-efficiency math.</strong> g5.xlarge runs at $0.45/hr Spot vs g4dn.xlarge at
$0.22/hr Spot &mdash; about 2&times; the cost. For 3.4&times; the throughput, that's
<strong>1.7&times; better cost efficiency</strong>, before any FP8 calibration work. With
calibrated FP8 (if quality holds), the advantage would grow further because the 4&times; KV
capacity would let you push concurrency past where the T4 cliff starts. The popular intuition
that "Spot T4 is the cheapest GPU you can run vLLM on" turns out to be wrong on a per-token
basis once you do the math &mdash; A10G wins on $/token, even at 2&times; the hourly rate.
</div>

<h3>What this section actually proved</h3>

<p>The conclusion is <em>not</em> "FP8 KV cache is broken." It's that <strong>FP8 KV cache is
a hardware-gated <em>and</em> calibration-gated optimization</strong>. Both gates have to be
open for it to work in production:</p>

<ol>
  <li><strong>Hardware gate:</strong> you need Ampere or newer (sm_80+). E5M2 on Turing is the
  wrong format on the wrong silicon &mdash; latency and throughput look fine, but the model is
  effectively broken for any output longer than a sentence.</li>
  <li><strong>Calibration gate:</strong> even on supported hardware with E4M3, the bare CLI
  flag isn't enough. Production deployments need an offline calibration pass (e.g.,
  <code>llm-compressor</code>) to compute per-channel scaling factors, then load them at
  serve time with <code>--quantization-param-path</code>.</li>
</ol>

<p>The way I'd think about this in production terms: the bare flag is a footgun, the mechanism
is sound. Anthropic-tier inference systems would never ship FP8 KV cache as "flip the flag and
see what happens" &mdash; they'd treat it the way Anthropic and the broader inference community
treat weight quantization, with calibration data, evals, and per-model validation before
production rollout. This is squarely on the Phase 5+ inference-optimization roadmap, on hardware
that supports it correctly.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-10"><span class="section-num">10</span> The cross-cutting finding: vLLM V1 is autonomous</h2>

<p>Halfway through the lab a pattern emerged: several knobs from V0-era documentation were
either silently ignored or rejected outright. vLLM v0.17.1 ships only the V1 engine, and V1
is a <strong>fully autonomous scheduler</strong>.</p>

<table>
  <thead>
    <tr><th>Flag</th><th>V1 behavior</th><th>Lab impact</th></tr>
  </thead>
  <tbody>
    <tr><td><code>--no-enable-prefix-caching</code></td><td>Works</td><td>Test 9 ran fine</td></tr>
    <tr><td><code>--no-enable-chunked-prefill</code></td><td>Accepted, ignored</td><td>ON/OFF identical</td></tr>
    <tr><td><code>--max-num-batched-tokens</code></td><td>Accepted, ignored</td><td>512/2048/4096 identical</td></tr>
    <tr><td><code>--preemption-mode</code></td><td>Rejected</td><td>V1 chose its own policy</td></tr>
    <tr><td><code>--speculative-model</code></td><td>Rejected</td><td>Couldn't run spec-decoding</td></tr>
  </tbody>
</table>

<div class="callout callout-info">
<strong>The autonomy is good, until it isn't.</strong> V1's chunked prefill produced a 1.02&times;
TBT spike under prefill bombs &mdash; better than any manually-tuned setting could plausibly
achieve. Auto-preemption handled 42 concurrent requests on a 34-max server (1.24&times; overload)
with 42/42 successes. The price is loss of control: there's no path to speculative decoding,
no way to override preemption strategy. This is the AWS Managed Auto Scaling tradeoff &mdash;
target tracking is usually right, until you need step scaling with custom alarms and
you can't get there.
</div>

<h3>What carries to the production stack</h3>
<ul>
  <li><strong>The KV cache is the master constraint.</strong> Concurrency, latency, throughput,
  and quality all bend around it. Every Phase 3+ scaling decision targets it.</li>
  <li><strong>Continuous batching is engine-level; quantization is a memory tradeoff.</strong>
  AWQ INT4 is the default for T4-class hardware: same quality, 60% faster, 2&times; KV headroom.</li>
  <li><strong>Prefix caching is the hidden multiplier.</strong> RAG, multi-turn, system prompts
  all lean on it. Cache-aware routing (Phase 5) exists to <em>preserve</em> hit rate across
  pod restarts and scale-outs.</li>
  <li><strong>vLLM does not protect itself.</strong> Every production cliff in this lab ended
  in a crash, not a 503. Phase 7 hardening (gateway admission, priority queues, SLO-aware
  early rejection from the Mooncake pattern) directly addresses this.</li>
  <li><strong>FP8 KV cache needs calibration, not just hardware.</strong> The Phase 5+ inference
  optimization track is where this gets a proper second pass &mdash; with <code>llm-compressor</code>
  and per-channel scaling factors, on hardware that supports E4M3 natively.</li>
</ul>

<p>Stages 1 and 2 produced no production system. They produced a lens. Every phase that comes
next &mdash; Karpenter, KEDA, admission control, cache-aware routing, disaggregated inference &mdash;
is a response to something visible in one of these charts.</p>
""")

    # ============================================================
    parts.append("""
<h2 id="sec-11"><span class="section-num">11</span> Hardware &amp; environment</h2>
<table>
  <thead>
    <tr><th>Component</th><th>Stage 1</th><th>Stage 2 (T4)</th><th>Stage 2 (A10G)</th></tr>
  </thead>
  <tbody>
    <tr><td>Hardware</td><td>Apple M4, 16GB unified</td><td>g4dn.xlarge, NVIDIA T4 16GB</td><td>g5.xlarge, NVIDIA A10G 24GB</td></tr>
    <tr><td>Compute capability</td><td>Apple Silicon</td><td>sm_75 (Turing)</td><td>sm_86 (Ampere)</td></tr>
    <tr><td>Memory bandwidth</td><td>~100 GB/s</td><td>320 GB/s</td><td>600 GB/s</td></tr>
    <tr><td>Engine</td><td>Ollama (llama.cpp)</td><td>vLLM v0.17.1 (V1)</td><td>vLLM v0.17.1 (V1)</td></tr>
    <tr><td>Model</td><td>Qwen2.5:7B Q4_K_M</td><td>Qwen2.5-7B-AWQ / GPTQ-Int8</td><td>Qwen2.5-7B-AWQ</td></tr>
    <tr><td>FP8 format supported</td><td>n/a</td><td>E5M2 only</td><td>E4M3 (native tensor cores)</td></tr>
    <tr><td>Spot cost</td><td>n/a (local)</td><td>~$0.22/hr</td><td>~$0.45/hr</td></tr>
  </tbody>
</table>

<p class="round-meta">Cold start (T4, EBS-cached weights): ~95s docker run &rarr; first response.
Breakdown: ~30s weight I/O, ~12s torch.compile, ~11s CUDA graph capture. This is the number
that drives the Phase 4 cold-start optimization track and Phase 1 Karpenter provisioning targets.</p>
""")

    parts.append("""
</div>

<div class="footer">
  LLM Inference Learning Lab &mdash; Stages 1 &amp; 2 &bull;
  Mohamed Mahmoud &bull;
  30 experiments &bull;
  Ollama on Apple M4 + vLLM v0.17.1 on g4dn.xlarge / g5.xlarge Spot &bull;
  Generated 2026-04-28
</div>

</body>
</html>
""")
    return "".join(parts)


HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Inference Learning Lab &mdash; Stages 1 &amp; 2</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
    background: #0d1117; color: #e6edf3; padding: 0; margin: 0;
  }
  .hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1c2128 100%);
    color: #e6edf3; padding: 48px 40px 40px; border-bottom: 3px solid #58a6ff;
  }
  .hero h1 { font-size: 32px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }
  .hero .subtitle { color: #8b949e; font-size: 15px; margin-bottom: 20px; max-width: 900px; line-height: 1.6; }
  .verdict-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px; }
  .verdict-chip {
    background: rgba(126,231,135,0.12); border: 1px solid rgba(126,231,135,0.3);
    border-radius: 8px; padding: 14px 20px; min-width: 220px; max-width: 320px;
  }
  .verdict-chip .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 4px; }
  .verdict-chip .value { font-size: 16px; font-weight: 700; color: #7ee787; line-height: 1.4; }
  .verdict-chip .value.blue { color: #58a6ff; }
  .verdict-chip .value.yellow { color: #e3b341; }
  .verdict-chip .value.purple { color: #d2a8ff; }
  .verdict-chip .value.cyan { color: #79c0ff; }
  .verdict-chip .detail { font-size: 12px; color: #8b949e; margin-top: 6px; line-height: 1.5; }

  .content { max-width: 1100px; margin: 0 auto; padding: 32px 40px 60px; }
  h2 { font-size: 22px; font-weight: 700; margin: 44px 0 14px; color: #e6edf3; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
  h3 { font-size: 17px; font-weight: 600; margin: 28px 0 12px; color: #58a6ff; }
  p { font-size: 15px; line-height: 1.75; color: #c9d1d9; margin-bottom: 14px; }
  li { font-size: 15px; line-height: 1.75; color: #c9d1d9; margin-bottom: 8px; }
  .round-meta { font-size: 13px; color: #8b949e; margin-bottom: 8px; }
  ul { padding-left: 24px; margin-bottom: 14px; }
  strong { color: #e6edf3; }
  em { color: #d2a8ff; font-style: italic; }
  code { font-family: 'SF Mono', monospace; font-size: 13px; background: #161b22; padding: 1px 6px; border-radius: 4px; color: #79c0ff; border: 1px solid #21262d; }

  figure.chart {
    margin: 22px 0; text-align: center;
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
  figure.chart img { max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 4px; }

  table { width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 13px; }
  th {
    text-align: left; padding: 10px 12px; font-weight: 600; color: #8b949e;
    border-bottom: 2px solid #30363d; background: #161b22; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.4px;
  }
  td { padding: 9px 12px; border-bottom: 1px solid #21262d; font-size: 13px; color: #c9d1d9; }
  td code { font-size: 12px; }

  .callout {
    border-radius: 8px; padding: 16px 20px; margin: 18px 0; font-size: 14px; line-height: 1.7;
    border-left: 4px solid; color: #c9d1d9;
  }
  .callout-info { background: rgba(88,166,255,0.08); border-color: #58a6ff; }
  .callout-info strong { color: #79c0ff; }
  .callout-success { background: rgba(63,185,80,0.08); border-color: #3fb950; }
  .callout-success strong { color: #7ee787; }
  .callout-warning { background: rgba(227,179,65,0.08); border-color: #e3b341; }
  .callout-warning strong { color: #f0c674; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
  .two-col p { font-size: 14px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }

  .section-num {
    display: inline-block; background: #0969da; color: #fff; min-width: 28px; height: 28px;
    border-radius: 50%; text-align: center; line-height: 28px; font-size: 13px;
    font-weight: 700; margin-right: 8px; vertical-align: middle; padding: 0 6px;
  }

  .footer { text-align: center; padding: 32px; color: #484f58; font-size: 12px; border-top: 1px solid #21262d; margin-top: 40px; line-height: 1.7; }

  .toc {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 22px 28px; margin: 8px 0 28px;
  }
  .toc-title {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #8b949e; margin-bottom: 14px; font-weight: 600;
  }
  .toc-list {
    list-style: none; padding: 0; margin: 0;
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px;
    counter-reset: toc;
  }
  .toc-list li {
    counter-increment: toc;
    font-size: 14px; line-height: 1.5; padding: 6px 0;
    border-bottom: 1px solid #21262d; margin: 0;
  }
  .toc-list li::before {
    content: counter(toc) ". ";
    color: #58a6ff; font-weight: 700; margin-right: 4px;
  }
  .toc-list a {
    color: #c9d1d9; text-decoration: none; font-weight: 500;
  }
  .toc-list a:hover { color: #79c0ff; text-decoration: underline; }
  .toc-hint {
    display: block; font-size: 12px; color: #8b949e; margin-top: 2px;
    margin-left: 18px;
  }
  @media (max-width: 800px) { .toc-list { grid-template-columns: 1fr; } }
  html { scroll-behavior: smooth; }
  h2[id] { scroll-margin-top: 16px; }
</style>
</head>
<body>
"""

TOC = """
<nav class="toc" aria-label="Section index">
  <div class="toc-title">Jump to a section</div>
  <ol class="toc-list">
    <li><a href="#sec-1">The question that started everything</a><span class="toc-hint">Sequential vs. continuous batching</span></li>
    <li><a href="#sec-2">How far can one GPU actually go?</a><span class="toc-hint">Concurrency curves &amp; the throughput knee</span></li>
    <li><a href="#sec-3">Quantization: shrinking the model</a><span class="toc-hint">AWQ vs. GPTQ &mdash; speed and quality</span></li>
    <li><a href="#sec-4">The KV cache is a CDN</a><span class="toc-hint">Prefix caching &amp; multi-turn</span></li>
    <li><a href="#sec-5">The wall, and what's on the other side</a><span class="toc-hint">Pushing KV to 100% &amp; the crash</span></li>
    <li><a href="#sec-6">Prefill and decode are not the same</a><span class="toc-hint">Why output tokens cost ~17&times; more</span></li>
    <li><a href="#sec-7">Two speed-neutral knobs</a><span class="toc-hint">Guided JSON &amp; sampling temperature</span></li>
    <li><a href="#sec-8">Where the production cliff actually is</a><span class="toc-hint">Tail latency, soak, degradation, prod sim</span></li>
    <li><a href="#sec-9">FP8 KV cache: hardware-gated</a><span class="toc-hint">T4 E5M2 vs. A10G E4M3</span></li>
    <li><a href="#sec-10">vLLM V1 is autonomous</a><span class="toc-hint">The cross-cutting engine finding</span></li>
    <li><a href="#sec-11">Hardware &amp; environment</a><span class="toc-hint">Reference table &amp; cold-start numbers</span></li>
  </ol>
</nav>
"""


HERO = """
<div class="hero">
  <h1>LLM Inference Learning Lab &mdash; Stages 1 &amp; 2</h1>
  <p class="subtitle">
    30 experiments across two engines and two GPU generations, designed to make a single
    distributed-systems leader fluent in inference infrastructure. Ollama on Apple M4 &rarr;
    vLLM v0.17.1 on AWS T4 and A10G Spot. From "what does sequential mean?" to FP8 KV cache
    on Ampere &mdash; one chart at a time.
  </p>
  <div class="verdict-bar">
    <div class="verdict-chip">
      <div class="label">The Question</div>
      <div class="value yellow">What does production inference actually look like under the hood?</div>
    </div>
    <div class="verdict-chip">
      <div class="label">The Throughline</div>
      <div class="value">The KV cache is the master constraint</div>
      <div class="detail">Concurrency, tail latency, prefix reuse, FP8 &mdash; everything bends around it.</div>
    </div>
    <div class="verdict-chip">
      <div class="label">The Surprise</div>
      <div class="value cyan">vLLM does not protect itself</div>
      <div class="detail">Every cliff ended in a crash, not a 503. External admission control is mandatory.</div>
    </div>
  </div>
</div>
"""


def main():
    nb = json.loads(NB_PATH.read_text())
    images = extract_images(nb)
    print(f"Found {len(images)} chart images in notebook")
    html = build_html(images)
    OUT.write_text(html)
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
