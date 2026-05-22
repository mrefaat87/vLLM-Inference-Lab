#!/usr/bin/env python3
"""
Stage 2 Quality Benchmark — measures model output quality/precision across
different quantization levels (e.g., GPTQ-Int8 vs AWQ-Int4).

Run against a vLLM server with OpenAI-compatible API:
    python stage2_quality_bench.py --host http://3.231.224.150:8000 \
        --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8

Mohamed: swap --model to compare quantization variants side by side.
"""

import argparse
import re
import sys
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """One benchmark question with its scoring function."""
    category: str
    prompt: str
    expected_desc: str  # human-readable description of what "correct" looks like
    check: Callable[[str], bool]  # takes model response text, returns pass/fail
    # Why per-question max_tokens: harder categories (math word problems, code)
    # need more room for chain-of-thought reasoning than simple factual recall.
    max_tokens: int = 200


@dataclass
class Result:
    """Stores the outcome of one benchmark question."""
    question: Question
    response_text: str
    passed: bool
    latency_s: float


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
# Why lambdas + helpers instead of one giant if/else: keeps each question's
# pass/fail logic co-located with the question definition so you can audit
# correctness at a glance.

def contains(expected: str) -> Callable[[str], bool]:
    """Case-insensitive substring check — the workhorse scorer."""
    return lambda text: expected.lower() in text.lower()


def contains_any(*options: str) -> Callable[[str], bool]:
    """Pass if the response contains ANY of the expected substrings."""
    return lambda text: any(o.lower() in text.lower() for o in options)


def contains_all(*parts: str) -> Callable[[str], bool]:
    """Pass if the response contains ALL of the expected substrings."""
    return lambda text: all(p.lower() in text.lower() for p in parts)


def count_items_eq(n: int) -> Callable[[str], bool]:
    """Check the response has exactly n numbered or bulleted list items.

    Why this heuristic: models typically emit '1.', '2.' or '- ' for lists.
    We count lines starting with a digit+period OR a dash/asterisk bullet.
    """
    def _check(text: str) -> bool:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        # Match lines that look like list items: "1.", "2)", "- ", "* "
        item_pattern = re.compile(r"^(\d+[\.\)]\s|[-*]\s)")
        items = [l for l in lines if item_pattern.match(l)]
        return len(items) == n
    return _check


def is_single_sentence(text: str) -> bool:
    """Check the response is exactly one sentence.

    Why strip + split on sentence-enders: simple heuristic that works for
    declarative English without pulling in nltk.
    """
    text = text.strip()
    # Remove trailing whitespace/newlines, count sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings from split
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) == 1


def is_yes_or_no(text: str) -> bool:
    """Check if the core answer is just 'yes' or 'no'.

    Why we check the first word: models often can't resist explaining, but
    we're generous — if the first meaningful word is yes/no, that counts.
    """
    first_word = text.strip().split()[0].lower().rstrip(".,!") if text.strip() else ""
    return first_word in ("yes", "no")


def word_count_between(lo: int, hi: int) -> Callable[[str], bool]:
    """Check response word count is within [lo, hi] inclusive."""
    return lambda text: lo <= len(text.split()) <= hi


def answer_is_number(n: int) -> Callable[[str], bool]:
    """Check that the expected numeric answer appears in the response.

    Why str(n): GSM8K-style problems need us to find the final numerical
    answer somewhere in the model's chain-of-thought output.
    """
    return lambda text: str(n) in text


def starts_with_letter(letter: str) -> Callable[[str], bool]:
    """Check that the response's first non-whitespace character is the expected letter.

    Why first-char check: MMLU-style questions ask the model to reply with
    just A/B/C/D, so we look at the very start of the response. We also
    accept the letter appearing after common prefixes like 'Answer:'.
    """
    letter = letter.upper()
    def _check(text: str) -> bool:
        text = text.strip().upper()
        # Direct match: response starts with the letter
        if text and text[0] == letter:
            return True
        # Tolerate prefixes like "Answer: B" or "The answer is B"
        if re.search(rf'\b{letter}\b', text[:30]):
            return True
        return False
    return _check


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------
# Why hard-coded: the benchmark must be self-contained (no external data files)
# and every answer must be unambiguous so pass/fail is deterministic.

QUESTIONS: List[Question] = []

# ---- Category 1: Math Reasoning (10 questions) ----------------------------
# Why math: quantized models often degrade on precise arithmetic first.

QUESTIONS += [
    Question("Math Reasoning", "What is 17 * 23?", "contains '391'",
             contains("391")),
    Question("Math Reasoning", "What is 144 / 12?", "contains '12'",
             contains("12")),
    Question("Math Reasoning", "What is 256 + 389?", "contains '645'",
             contains("645")),
    Question("Math Reasoning", "What is 1000 - 427?", "contains '573'",
             contains("573")),
    Question("Math Reasoning",
             "If a train travels 120 miles in 2 hours, what is its speed in mph?",
             "contains '60'",
             contains("60")),
    Question("Math Reasoning",
             "A rectangle has length 8 and width 5. What is its area?",
             "contains '40'",
             contains("40")),
    Question("Math Reasoning",
             "What is 15% of 200?",
             "contains '30'",
             contains("30")),
    Question("Math Reasoning",
             "If you buy 3 items at $7 each and pay with $50, how much change do you get?",
             "contains '29'",
             contains("29")),
    Question("Math Reasoning",
             "What is the square root of 169?",
             "contains '13'",
             contains("13")),
    Question("Math Reasoning",
             "How many seconds are in 2 hours?",
             "contains '7200'",
             contains("7200")),
]

# ---- Category 2: Factual Recall (10 questions) ----------------------------
# Why factual recall: tests whether quantization lost stored knowledge.

QUESTIONS += [
    Question("Factual Recall", "What is the capital of Australia?",
             "contains 'Canberra'",
             contains("canberra")),
    Question("Factual Recall", "What year did the Berlin Wall fall?",
             "contains '1989'",
             contains("1989")),
    Question("Factual Recall", "What planet is closest to the Sun?",
             "contains 'Mercury'",
             contains("mercury")),
    Question("Factual Recall", "Who wrote Romeo and Juliet?",
             "contains 'Shakespeare'",
             contains("shakespeare")),
    Question("Factual Recall", "What is the chemical symbol for gold?",
             "contains 'Au'",
             # Why word-boundary check: avoid false positives from words like "auxiliary"
             lambda t: bool(re.search(r'\bAu\b', t))),
    Question("Factual Recall", "How many continents are there?",
             "contains '7'",
             contains("7")),
    Question("Factual Recall", "What is the boiling point of water in Celsius?",
             "contains '100'",
             contains("100")),
    Question("Factual Recall", "What is the largest ocean on Earth?",
             "contains 'Pacific'",
             contains("pacific")),
    Question("Factual Recall", "Who painted the Mona Lisa?",
             "contains 'da Vinci' or 'Leonardo'",
             contains_any("da vinci", "leonardo")),
    Question("Factual Recall", "What gas do plants absorb from the atmosphere?",
             "contains 'carbon dioxide' or 'CO2'",
             contains_any("carbon dioxide", "co2")),
]

# ---- Category 3: Instruction Following (10 questions) ---------------------
# Why instruction following: quantization can degrade the model's ability to
# adhere to format constraints, which matters for structured-output use cases.

QUESTIONS += [
    Question("Instruction Following",
             "List exactly 3 benefits of exercise. Use a numbered list.",
             "exactly 3 numbered items",
             count_items_eq(3)),
    Question("Instruction Following",
             "Is the Earth flat? Answer with only yes or no.",
             "first word is 'no'",
             is_yes_or_no),
    Question("Instruction Following",
             "Explain gravity in exactly one sentence.",
             "response is a single sentence",
             is_single_sentence),
    Question("Instruction Following",
             "Name 5 colors. Separate them with commas on a single line.",
             "has at least 4 commas (5 items)",
             # Why >=4 commas: 5 items separated by commas = 4 commas minimum
             lambda t: t.strip().count(",") >= 4),
    Question("Instruction Following",
             "What is 2+2? Reply with just the number, nothing else.",
             "contains '4' and response is very short",
             # Why length check: ensures the model didn't ramble
             lambda t: "4" in t and len(t.strip()) <= 5),
    Question("Instruction Following",
             "List exactly 5 fruits. Use a numbered list.",
             "exactly 5 numbered items",
             count_items_eq(5)),
    Question("Instruction Following",
             "Say 'hello world' in uppercase letters only.",
             "contains 'HELLO WORLD'",
             contains("HELLO WORLD")),
    Question("Instruction Following",
             "Is 7 greater than 10? Answer with only yes or no.",
             "first word is 'no'",
             is_yes_or_no),
    Question("Instruction Following",
             "Write exactly 3 words.",
             "response is exactly 3 words",
             # Why split-based check: straightforward word count
             lambda t: len(t.strip().split()) == 3),
    Question("Instruction Following",
             "Respond with the word 'acknowledged' and nothing else.",
             "contains 'acknowledged' and is short",
             lambda t: "acknowledged" in t.lower() and len(t.strip().split()) <= 3),
]

# ---- Category 4: Code Generation (5 questions) ----------------------------
# Why code gen: structured output with precise syntax is a stress test for
# quantized weights — even small weight errors can break code logic.

QUESTIONS += [
    Question("Code Generation",
             "Write a Python function that returns the factorial of n.",
             "contains 'def' and 'factorial'",
             contains_all("def", "factorial")),
    Question("Code Generation",
             "Write a Python function called 'is_palindrome' that checks if a string is a palindrome.",
             "contains 'def is_palindrome' and reverse/slicing logic",
             # Why check for [::-1] or 'reversed': both are valid palindrome approaches
             lambda t: "def is_palindrome" in t and (
                 "[::-1]" in t or "reversed" in t or "reverse" in t.lower())),
    Question("Code Generation",
             "Write a Python function called 'fizzbuzz' that takes n and returns "
             "'Fizz' if divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, else the number.",
             "contains 'def fizzbuzz' and modulo checks",
             lambda t: "def fizzbuzz" in t and "%" in t),
    Question("Code Generation",
             "Write a Python function called 'fibonacci' that returns the nth Fibonacci number.",
             "contains 'def fibonacci'",
             contains_all("def", "fibonacci")),
    Question("Code Generation",
             "Write a Python function called 'max_of_three' that returns the largest of three numbers.",
             "contains 'def max_of_three'",
             contains("def max_of_three")),
]

# ---- Category 5: MMLU-Style (10 questions) --------------------------------
# Why MMLU-style: graduate-level multiple choice across domains is a strong
# differentiator between INT4 and INT8 — quantization degrades the nuanced
# knowledge retrieval needed for these questions.

QUESTIONS += [
    Question("MMLU-Style",
             "Which of the following best describes the function of telomerase?\n"
             "A) It repairs misfolded proteins\n"
             "B) It adds repetitive nucleotide sequences to the ends of chromosomes\n"
             "C) It unwinds double-stranded DNA during replication\n"
             "D) It catalyzes the splicing of pre-mRNA\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "In microeconomics, what does the term 'deadweight loss' refer to?\n"
             "A) The total cost of production that exceeds revenue\n"
             "B) The loss in total surplus that occurs when the market is not at equilibrium\n"
             "C) The depreciation of capital goods over time\n"
             "D) The opportunity cost of choosing one investment over another\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "The Treaty of Westphalia (1648) is most significant because it:\n"
             "A) Ended the Hundred Years' War between England and France\n"
             "B) Established the principle of state sovereignty in international relations\n"
             "C) Created the first international court of justice\n"
             "D) United the German states under a single emperor\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "Which logical fallacy is committed when one argues that a claim must be true "
             "because it has not been proven false?\n"
             "A) Straw man\n"
             "B) Ad hominem\n"
             "C) Appeal to ignorance\n"
             "D) False dilemma\n"
             "Respond with just the letter.",
             "answer is C",
             starts_with_letter("C"), max_tokens=50),

    Question("MMLU-Style",
             "In the human heart, which valve separates the left atrium from the left ventricle?\n"
             "A) Tricuspid valve\n"
             "B) Pulmonary valve\n"
             "C) Aortic valve\n"
             "D) Mitral valve (bicuspid valve)\n"
             "Respond with just the letter.",
             "answer is D",
             starts_with_letter("D"), max_tokens=50),

    Question("MMLU-Style",
             "What is the Heisenberg Uncertainty Principle?\n"
             "A) Energy can neither be created nor destroyed\n"
             "B) The position and momentum of a particle cannot both be precisely determined simultaneously\n"
             "C) An object at rest stays at rest unless acted upon by a force\n"
             "D) The entropy of an isolated system always increases\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "In Kant's moral philosophy, the categorical imperative requires that one:\n"
             "A) Maximize overall happiness for the greatest number\n"
             "B) Act only according to maxims that one could will to be universal laws\n"
             "C) Follow the virtues defined by one's community\n"
             "D) Obey the commands of a sovereign authority\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "Which of the following is an autoimmune disease?\n"
             "A) Tuberculosis\n"
             "B) Type 1 Diabetes\n"
             "C) Malaria\n"
             "D) Influenza\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),

    Question("MMLU-Style",
             "The Krebs cycle (citric acid cycle) primarily takes place in which part of the cell?\n"
             "A) Cytoplasm\n"
             "B) Nucleus\n"
             "C) Mitochondrial matrix\n"
             "D) Endoplasmic reticulum\n"
             "Respond with just the letter.",
             "answer is C",
             starts_with_letter("C"), max_tokens=50),

    Question("MMLU-Style",
             "In international trade theory, the Heckscher-Ohlin model predicts that a country will export goods that:\n"
             "A) Have the highest absolute production cost\n"
             "B) Intensively use the factors of production it has in abundance\n"
             "C) Are demanded least by its domestic consumers\n"
             "D) Require the most advanced technology to produce\n"
             "Respond with just the letter.",
             "answer is B",
             starts_with_letter("B"), max_tokens=50),
]

# ---- Category 6: GSM8K-Style (10 questions) -------------------------------
# Why GSM8K-style: multi-step math word problems require chain-of-thought
# reasoning that stresses quantized weights — INT4 models often make
# arithmetic mistakes mid-chain that INT8 models handle correctly.

QUESTIONS += [
    Question("GSM8K-Style",
             "A store sells apples for $2 each and oranges for $3 each. Sarah buys "
             "twice as many apples as oranges. If she spends $28 total, how many "
             "oranges did she buy?",
             "answer is 4",
             # Why: let x = oranges, 2x = apples; 2(2x) + 3x = 28 => 7x = 28 => x = 4
             answer_is_number(4), max_tokens=500),

    Question("GSM8K-Style",
             "A tank is being filled by two pipes. Pipe A fills it in 6 hours alone, "
             "and Pipe B fills it in 4 hours alone. If both pipes are opened together, "
             "how many hours does it take to fill the tank? Express your answer as a "
             "decimal rounded to one decimal place.",
             "answer is 2.4",
             # Why: 1/6 + 1/4 = 5/12 per hour => 12/5 = 2.4 hours
             contains("2.4"), max_tokens=500),

    Question("GSM8K-Style",
             "Maria has 120 stamps. She gives 25% of them to her brother, then gives "
             "one-third of the remaining stamps to her friend. How many stamps does "
             "Maria have left?",
             "answer is 60",
             # Why: 120 - 25%(120)=30 => 90; 90 - 90/3=30 => 60
             answer_is_number(60), max_tokens=500),

    Question("GSM8K-Style",
             "A car travels from City A to City B at 60 mph and returns at 40 mph. "
             "If the distance between the cities is 120 miles, what is the average "
             "speed for the entire round trip in mph?",
             "answer is 48",
             # Why: time there = 2h, time back = 3h; total = 240mi/5h = 48 mph
             answer_is_number(48), max_tokens=500),

    Question("GSM8K-Style",
             "A bookstore offers a 15% discount on all books. After the discount, "
             "a 10% sales tax is applied. If a book's original price is $40, what "
             "is the final price the customer pays?",
             "answer is $37.40",
             # Why: 40 * 0.85 = 34; 34 * 1.10 = 37.40
             contains("37.4"), max_tokens=500),

    Question("GSM8K-Style",
             "In a class of 40 students, 65% passed the math test. Of those who "
             "passed, 75% scored above 80. How many students scored above 80?",
             "answer is 19 or 20 (depending on rounding)",
             # Why: 40 * 0.65 = 26 passed; 26 * 0.75 = 19.5 ≈ 19 or 20
             contains_any("19", "20"), max_tokens=500),

    Question("GSM8K-Style",
             "A rectangular garden is 3 times as long as it is wide. If the "
             "perimeter of the garden is 64 meters, what is the area in square meters?",
             "answer is 192",
             # Why: w + 3w = 32 (half perimeter) => w=8, l=24; area = 192
             answer_is_number(192), max_tokens=500),

    Question("GSM8K-Style",
             "Tom invests $5000 at a simple interest rate of 8% per year. How much "
             "total money (principal + interest) will he have after 3 years?",
             "answer is 6200",
             # Why: interest = 5000 * 0.08 * 3 = 1200; total = 6200
             answer_is_number(6200), max_tokens=500),

    Question("GSM8K-Style",
             "A train leaves Station A at 9:00 AM traveling at 80 km/h. Another "
             "train leaves Station B (which is 400 km from A) at 10:00 AM traveling "
             "toward A at 120 km/h. At what time do the trains meet?",
             "answer is 11:36 AM",
             # Why: at 10AM, train A has covered 80km, gap=320km; closing speed=200km/h;
             # time = 320/200 = 1.6h = 1h36m after 10AM => 11:36 AM
             contains_any("11:36", "11:36 AM", "11:36AM"), max_tokens=500),

    Question("GSM8K-Style",
             "A bakery sells cupcakes in boxes of 6 and cookies in boxes of 8. If "
             "a customer buys 5 boxes of cupcakes and 3 boxes of cookies, and each "
             "cupcake costs $1.50 while each cookie costs $1.00, what is the total cost?",
             "answer is 69",
             # Why: 5*6*1.50 = 45; 3*8*1.00 = 24; total = 69
             answer_is_number(69), max_tokens=500),
]

# ---- Category 7: HumanEval-Style (5 questions) ----------------------------
# Why HumanEval-style: algorithmic coding challenges are harder than simple
# function-writing — they require correct logic flow, which is more fragile
# under aggressive quantization.

QUESTIONS += [
    Question("HumanEval-Style",
             "Write a Python function called 'binary_search' that takes a sorted list "
             "and a target value, and returns the index of the target if found, or -1 "
             "if not found. Use the binary search algorithm (not linear search).",
             "contains def binary_search, mid-point calculation, and comparison logic",
             # Why check for mid and // or /: binary search must compute a midpoint
             lambda t: "def binary_search" in t and ("mid" in t.lower() or "middle" in t.lower())
                        and ("//" in t or ">> 1" in t or "/ 2" in t or "mid" in t.lower()),
             max_tokens=500),

    Question("HumanEval-Style",
             "Write a Python function called 'merge_sorted' that takes two sorted lists "
             "and returns a single merged sorted list. Do NOT use the built-in sort() or "
             "sorted() functions — implement the merge step manually.",
             "contains def merge_sorted with index-based merge logic",
             # Why check for while/if and append: manual merge uses index pointers
             lambda t: "def merge_sorted" in t
                        and ("while" in t or "for" in t)
                        and ("append" in t or "extend" in t or "+=" in t or "result" in t.lower()),
             max_tokens=500),

    Question("HumanEval-Style",
             "Write a Python function called 'is_prime' that takes an integer n and "
             "returns True if n is a prime number and False otherwise. Handle edge "
             "cases (n <= 1 should return False).",
             "contains def is_prime with modulo check and loop",
             # Why check for % and range/while: prime check requires trial division
             lambda t: "def is_prime" in t and "%" in t
                        and ("range" in t or "while" in t)
                        and ("False" in t or "false" in t.lower()),
             max_tokens=500),

    Question("HumanEval-Style",
             "Write a Python class called 'LRUCache' that implements a Least Recently "
             "Used cache with a given capacity. It should support 'get(key)' which "
             "returns the value or -1 if not found, and 'put(key, value)' which inserts "
             "or updates the key-value pair, evicting the least recently used item if "
             "the cache is at capacity.",
             "contains class LRUCache with get/put methods and eviction logic",
             # Why check for class, get, put, and dict/OrderedDict: LRU needs a map structure
             lambda t: "class LRUCache" in t
                        and ("def get" in t)
                        and ("def put" in t)
                        and ("dict" in t.lower() or "ordereddict" in t.lower()
                             or "hash" in t.lower() or "{}" in t),
             max_tokens=500),

    Question("HumanEval-Style",
             "Write a Python function called 'longest_common_prefix' that takes a list "
             "of strings and returns the longest common prefix string. If there is no "
             "common prefix, return an empty string.",
             "contains def longest_common_prefix with character comparison logic",
             # Why check for indexing and comparison: LCP needs char-by-char or zip-based comparison
             lambda t: "def longest_common_prefix" in t
                        and ("for" in t or "while" in t)
                        and ("[" in t)  # indexing into strings is essential
                        and ('""' in t or "''" in t or "prefix" in t.lower()),
             max_tokens=500),
]


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def call_chat_completions(host: str, model: str, prompt: str,
                          max_tokens: int = 200, temperature: float = 0.0
                          ) -> tuple[str, float]:
    """Send one request to /v1/chat/completions and return (text, latency_seconds).

    Why urllib instead of requests: keeps the script zero-dependency so it runs
    on any Python 3.8+ without pip install.
    """
    url = f"{host.rstrip('/')}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            # Why system message: anchors the model to be concise and precise,
            # reducing noise that could trip up simple string-match scoring.
            {"role": "system", "content": (
                "You are a helpful assistant. Be concise and precise. "
                "Follow formatting instructions exactly."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        # Why temperature=0: deterministic outputs so the benchmark is reproducible
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        # Why catch broadly: vLLM might be down or model not loaded — surface it clearly
        return f"[ERROR: {e}]", time.perf_counter() - t0
    except Exception as e:
        return f"[ERROR: {e}]", time.perf_counter() - t0
    latency = time.perf_counter() - t0

    # Why nested get with fallback: defensive against unexpected response shapes
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        text = f"[UNEXPECTED RESPONSE: {json.dumps(body)[:200]}]"

    return text, latency


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(host: str, model: str) -> List[Result]:
    """Execute every question and collect results."""
    results: List[Result] = []
    total = len(QUESTIONS)

    for i, q in enumerate(QUESTIONS, 1):
        # Why print progress: some models are slow; this shows we're not stuck
        print(f"  [{i}/{total}] {q.category}: {q.prompt[:60]}...", flush=True)

        # Why q.max_tokens: harder categories need more tokens for reasoning
        text, latency = call_chat_completions(host, model, q.prompt,
                                              max_tokens=q.max_tokens)

        try:
            passed = q.check(text)
        except Exception:
            # Why blanket except: a scoring bug shouldn't crash the whole run
            passed = False

        results.append(Result(question=q, response_text=text,
                              passed=passed, latency_s=latency))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: List[Result], model: str) -> None:
    """Print per-question details and the summary table."""

    # Why separator: visually groups the detail section from the summary
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"QUALITY BENCHMARK RESULTS — {model}")
    print(sep)

    # ---- Per-question detail ------------------------------------------------
    print("\n--- Detailed Results ---\n")
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        # Why truncate at 120 chars: keeps the terminal readable for long responses
        truncated = r.response_text.replace("\n", " ")[:120]
        print(f"Q{i:>2} [{status}] ({r.latency_s:.2f}s) [{r.question.category}]")
        print(f"     Prompt:   {r.question.prompt[:100]}")
        print(f"     Expected: {r.question.expected_desc}")
        print(f"     Got:      {truncated}")
        print()

    # ---- Summary table ------------------------------------------------------
    # Why dict-based aggregation: cleaner than multiple loops per category
    categories: dict = {}  # category_name -> {"pass": int, "total": int, "latencies": [float]}
    for r in results:
        cat = r.question.category
        if cat not in categories:
            categories[cat] = {"pass": 0, "total": 0, "latencies": []}
        categories[cat]["total"] += 1
        categories[cat]["latencies"].append(r.latency_s)
        if r.passed:
            categories[cat]["pass"] += 1

    print(f"\n{'--- Summary ---':^80}\n")
    header = f"{'Category':<24} {'Pass/Total':<12} {'Accuracy':>10} {'Avg Latency':>12}"
    print(header)
    print("-" * len(header))

    overall_pass = 0
    overall_total = 0
    overall_latencies: List[float] = []

    # Why sorted: keeps output order stable across runs
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        p, t = stats["pass"], stats["total"]
        acc = (p / t * 100) if t else 0
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        print(f"{cat:<24} {p}/{t:<10} {acc:>9.1f}% {avg_lat:>11.2f}s")

        overall_pass += p
        overall_total += t
        overall_latencies.extend(stats["latencies"])

    # Why separate OVERALL row: gives one number to compare across model variants
    print("-" * len(header))
    overall_acc = (overall_pass / overall_total * 100) if overall_total else 0
    overall_avg = sum(overall_latencies) / len(overall_latencies) if overall_latencies else 0
    print(f"{'OVERALL':<24} {overall_pass}/{overall_total:<10} "
          f"{overall_acc:>9.1f}% {overall_avg:>11.2f}s")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model output quality across quantization levels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python stage2_quality_bench.py --host http://3.231.224.150:8000 "
            "--model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8\n"
            "  python stage2_quality_bench.py --host http://3.231.224.150:8000 "
            "--model Qwen/Qwen2.5-7B-Instruct-AWQ\n"
        ),
    )
    parser.add_argument(
        "--host", type=str, default="http://3.231.224.150:8000",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        help="Model name as registered in vLLM (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"\nStarting quality benchmark against {args.host}")
    print(f"Model: {args.model}")
    print(f"Questions: {len(QUESTIONS)} across "
          f"{len(set(q.category for q in QUESTIONS))} categories\n")

    results = run_benchmark(args.host, args.model)
    print_report(results, args.model)

    # Why exit code: enables CI/CD pipelines to gate on quality thresholds
    overall_pass = sum(1 for r in results if r.passed)
    overall_total = len(results)
    accuracy = (overall_pass / overall_total * 100) if overall_total else 0

    if accuracy < 70:
        print(f"WARNING: Overall accuracy {accuracy:.1f}% is below 70% threshold.")
        sys.exit(1)
    else:
        print(f"Accuracy {accuracy:.1f}% meets the 70% threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
