#!/usr/bin/env python3
"""
Test 29C: FP8 KV Cache — Accuracy Preservation

Runs the same 60-question quality benchmark as Test 7 against an FP8 KV cache
server. Saves detailed JSON output with per-question response text for
side-by-side diff analysis against FP16 baseline.

Hypothesis: FP8 KV cache has zero measurable quality impact. The E5M2 format
loses precision in the 6th significant digit — far below what affects token
selection through softmax.

AWS analogy: Memory overcommit doesn't change the correctness of request
handling. The data path is identical; only the storage format changes.

Usage: python3 stage2_test29c_fp8_quality.py --host http://<ip>:8000
"""

import argparse
import re
import sys
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import List, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """One benchmark question with its scoring function."""
    category: str
    prompt: str
    expected_desc: str
    check: Callable[[str], bool]
    max_tokens: int = 200


@dataclass
class Result:
    """Stores the outcome of one benchmark question."""
    category: str
    prompt: str
    expected_desc: str
    response_text: str
    passed: bool
    latency_s: float


# ---------------------------------------------------------------------------
# Scoring helpers (identical to stage2_quality_bench.py)
# ---------------------------------------------------------------------------

def contains(expected: str) -> Callable[[str], bool]:
    return lambda text: expected.lower() in text.lower()

def contains_any(*options: str) -> Callable[[str], bool]:
    return lambda text: any(o.lower() in text.lower() for o in options)

def contains_all(*parts: str) -> Callable[[str], bool]:
    return lambda text: all(p.lower() in text.lower() for p in parts)

def count_items_eq(n: int) -> Callable[[str], bool]:
    def _check(text: str) -> bool:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        item_pattern = re.compile(r"^(\d+[\.\)]\s|[-*]\s)")
        items = [l for l in lines if item_pattern.match(l)]
        return len(items) == n
    return _check

def is_single_sentence(text: str) -> bool:
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) == 1

def is_yes_or_no(text: str) -> bool:
    first_word = text.strip().split()[0].lower().rstrip(".,!") if text.strip() else ""
    return first_word in ("yes", "no")

def word_count_between(lo: int, hi: int) -> Callable[[str], bool]:
    return lambda text: lo <= len(text.split()) <= hi

def answer_is_number(n: int) -> Callable[[str], bool]:
    return lambda text: str(n) in text

def starts_with_letter(letter: str) -> Callable[[str], bool]:
    letter = letter.upper()
    def _check(text: str) -> bool:
        text = text.strip().upper()
        if text and text[0] == letter:
            return True
        if re.search(rf'\b{letter}\b', text[:30]):
            return True
        return False
    return _check


# ---------------------------------------------------------------------------
# Question bank (identical to stage2_quality_bench.py)
# ---------------------------------------------------------------------------

QUESTIONS: List[Question] = []

# Category 1: Math Reasoning (10 questions)
QUESTIONS += [
    Question("Math Reasoning", "What is 17 * 23?", "contains '391'", contains("391")),
    Question("Math Reasoning", "What is 144 / 12?", "contains '12'", contains("12")),
    Question("Math Reasoning", "What is 256 + 389?", "contains '645'", contains("645")),
    Question("Math Reasoning", "What is 1000 - 427?", "contains '573'", contains("573")),
    Question("Math Reasoning", "If a train travels 120 miles in 2 hours, what is its speed in mph?", "contains '60'", contains("60")),
    Question("Math Reasoning", "A rectangle has length 8 and width 5. What is its area?", "contains '40'", contains("40")),
    Question("Math Reasoning", "What is 15% of 200?", "contains '30'", contains("30")),
    Question("Math Reasoning", "If you buy 3 items at $7 each and pay with $50, how much change do you get?", "contains '29'", contains("29")),
    Question("Math Reasoning", "What is the square root of 169?", "contains '13'", contains("13")),
    Question("Math Reasoning", "How many seconds are in 2 hours?", "contains '7200'", contains("7200")),
]

# Category 2: Factual Recall (10 questions)
QUESTIONS += [
    Question("Factual Recall", "What is the capital of Australia?", "contains 'Canberra'", contains("canberra")),
    Question("Factual Recall", "What year did the Berlin Wall fall?", "contains '1989'", contains("1989")),
    Question("Factual Recall", "What planet is closest to the Sun?", "contains 'Mercury'", contains("mercury")),
    Question("Factual Recall", "Who wrote Romeo and Juliet?", "contains 'Shakespeare'", contains("shakespeare")),
    Question("Factual Recall", "What is the chemical symbol for gold?", "contains 'Au'", lambda t: bool(re.search(r'\bAu\b', t))),
    Question("Factual Recall", "How many continents are there?", "contains '7'", contains("7")),
    Question("Factual Recall", "What is the boiling point of water in Celsius?", "contains '100'", contains("100")),
    Question("Factual Recall", "What is the largest ocean on Earth?", "contains 'Pacific'", contains("pacific")),
    Question("Factual Recall", "Who painted the Mona Lisa?", "contains 'da Vinci' or 'Leonardo'", contains_any("da vinci", "leonardo")),
    Question("Factual Recall", "What gas do plants absorb from the atmosphere?", "contains 'carbon dioxide' or 'CO2'", contains_any("carbon dioxide", "co2")),
]

# Category 3: Instruction Following (10 questions)
QUESTIONS += [
    Question("Instruction Following", "List exactly 3 benefits of exercise. Use a numbered list.", "exactly 3 numbered items", count_items_eq(3)),
    Question("Instruction Following", "Is the Earth flat? Answer with only yes or no.", "first word is 'no'", is_yes_or_no),
    Question("Instruction Following", "Explain gravity in exactly one sentence.", "response is a single sentence", is_single_sentence),
    Question("Instruction Following", "Name 5 colors. Separate them with commas on a single line.", "has at least 4 commas (5 items)", lambda t: t.strip().count(",") >= 4),
    Question("Instruction Following", "What is 2+2? Reply with just the number, nothing else.", "contains '4' and response is very short", lambda t: "4" in t and len(t.strip()) <= 5),
    Question("Instruction Following", "List exactly 5 fruits. Use a numbered list.", "exactly 5 numbered items", count_items_eq(5)),
    Question("Instruction Following", "Say 'hello world' in uppercase letters only.", "contains 'HELLO WORLD'", contains("HELLO WORLD")),
    Question("Instruction Following", "Is 7 greater than 10? Answer with only yes or no.", "first word is 'no'", is_yes_or_no),
    Question("Instruction Following", "Write exactly 3 words.", "response is exactly 3 words", lambda t: len(t.strip().split()) == 3),
    Question("Instruction Following", "Respond with the word 'acknowledged' and nothing else.", "contains 'acknowledged' and is short", lambda t: "acknowledged" in t.lower() and len(t.strip().split()) <= 3),
]

# Category 4: Code Generation (5 questions)
QUESTIONS += [
    Question("Code Generation", "Write a Python function that returns the factorial of n.", "contains 'def' and 'factorial'", contains_all("def", "factorial")),
    Question("Code Generation", "Write a Python function called 'is_palindrome' that checks if a string is a palindrome.", "contains 'def is_palindrome' and reverse/slicing logic", lambda t: "def is_palindrome" in t and ("[::-1]" in t or "reversed" in t or "reverse" in t.lower())),
    Question("Code Generation", "Write a Python function called 'fizzbuzz' that takes n and returns 'Fizz' if divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, else the number.", "contains 'def fizzbuzz' and modulo checks", lambda t: "def fizzbuzz" in t and "%" in t),
    Question("Code Generation", "Write a Python function called 'fibonacci' that returns the nth Fibonacci number.", "contains 'def fibonacci'", contains_all("def", "fibonacci")),
    Question("Code Generation", "Write a Python function called 'max_of_three' that returns the largest of three numbers.", "contains 'def max_of_three'", contains("def max_of_three")),
]

# Category 5: MMLU-Style (10 questions)
QUESTIONS += [
    Question("MMLU-Style", "Which of the following best describes the function of telomerase?\nA) It repairs misfolded proteins\nB) It adds repetitive nucleotide sequences to the ends of chromosomes\nC) It unwinds double-stranded DNA during replication\nD) It catalyzes the splicing of pre-mRNA\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "In microeconomics, what does the term 'deadweight loss' refer to?\nA) The total cost of production that exceeds revenue\nB) The loss in total surplus that occurs when the market is not at equilibrium\nC) The depreciation of capital goods over time\nD) The opportunity cost of choosing one investment over another\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "The Treaty of Westphalia (1648) is most significant because it:\nA) Ended the Hundred Years' War between England and France\nB) Established the principle of state sovereignty in international relations\nC) Created the first international court of justice\nD) United the German states under a single emperor\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "Which logical fallacy is committed when one argues that a claim must be true because it has not been proven false?\nA) Straw man\nB) Ad hominem\nC) Appeal to ignorance\nD) False dilemma\nRespond with just the letter.", "answer is C", starts_with_letter("C"), max_tokens=50),
    Question("MMLU-Style", "In the human heart, which valve separates the left atrium from the left ventricle?\nA) Tricuspid valve\nB) Pulmonary valve\nC) Aortic valve\nD) Mitral valve (bicuspid valve)\nRespond with just the letter.", "answer is D", starts_with_letter("D"), max_tokens=50),
    Question("MMLU-Style", "What is the Heisenberg Uncertainty Principle?\nA) Energy can neither be created nor destroyed\nB) The position and momentum of a particle cannot both be precisely determined simultaneously\nC) An object at rest stays at rest unless acted upon by a force\nD) The entropy of an isolated system always increases\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "In Kant's moral philosophy, the categorical imperative requires that one:\nA) Maximize overall happiness for the greatest number\nB) Act only according to maxims that one could will to be universal laws\nC) Follow the virtues defined by one's community\nD) Obey the commands of a sovereign authority\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "Which of the following is an autoimmune disease?\nA) Tuberculosis\nB) Type 1 Diabetes\nC) Malaria\nD) Influenza\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
    Question("MMLU-Style", "The Krebs cycle (citric acid cycle) primarily takes place in which part of the cell?\nA) Cytoplasm\nB) Nucleus\nC) Mitochondrial matrix\nD) Endoplasmic reticulum\nRespond with just the letter.", "answer is C", starts_with_letter("C"), max_tokens=50),
    Question("MMLU-Style", "In international trade theory, the Heckscher-Ohlin model predicts that a country will export goods that:\nA) Have the highest absolute production cost\nB) Intensively use the factors of production it has in abundance\nC) Are demanded least by its domestic consumers\nD) Require the most advanced technology to produce\nRespond with just the letter.", "answer is B", starts_with_letter("B"), max_tokens=50),
]

# Category 6: GSM8K-Style (10 questions)
QUESTIONS += [
    Question("GSM8K-Style", "A store sells apples for $2 each and oranges for $3 each. Sarah buys twice as many apples as oranges. If she spends $28 total, how many oranges did she buy?", "answer is 4", answer_is_number(4), max_tokens=500),
    Question("GSM8K-Style", "A tank is being filled by two pipes. Pipe A fills it in 6 hours alone, and Pipe B fills it in 4 hours alone. If both pipes are opened together, how many hours does it take to fill the tank? Express your answer as a decimal rounded to one decimal place.", "answer is 2.4", contains("2.4"), max_tokens=500),
    Question("GSM8K-Style", "Maria has 120 stamps. She gives 25% of them to her brother, then gives one-third of the remaining stamps to her friend. How many stamps does Maria have left?", "answer is 60", answer_is_number(60), max_tokens=500),
    Question("GSM8K-Style", "A car travels from City A to City B at 60 mph and returns at 40 mph. If the distance between the cities is 120 miles, what is the average speed for the entire round trip in mph?", "answer is 48", answer_is_number(48), max_tokens=500),
    Question("GSM8K-Style", "A bookstore offers a 15% discount on all books. After the discount, a 10% sales tax is applied. If a book's original price is $40, what is the final price the customer pays?", "answer is $37.40", contains("37.4"), max_tokens=500),
    Question("GSM8K-Style", "In a class of 40 students, 65% passed the math test. Of those who passed, 75% scored above 80. How many students scored above 80?", "answer is 19 or 20 (depending on rounding)", contains_any("19", "20"), max_tokens=500),
    Question("GSM8K-Style", "A rectangular garden is 3 times as long as it is wide. If the perimeter of the garden is 64 meters, what is the area in square meters?", "answer is 192", answer_is_number(192), max_tokens=500),
    Question("GSM8K-Style", "Tom invests $5000 at a simple interest rate of 8% per year. How much total money (principal + interest) will he have after 3 years?", "answer is 6200", answer_is_number(6200), max_tokens=500),
    Question("GSM8K-Style", "A train leaves Station A at 9:00 AM traveling at 80 km/h. Another train leaves Station B (which is 400 km from A) at 10:00 AM traveling toward A at 120 km/h. At what time do the trains meet?", "answer is 11:36 AM", contains_any("11:36", "11:36 AM", "11:36AM"), max_tokens=500),
    Question("GSM8K-Style", "A bakery sells cupcakes in boxes of 6 and cookies in boxes of 8. If a customer buys 5 boxes of cupcakes and 3 boxes of cookies, and each cupcake costs $1.50 while each cookie costs $1.00, what is the total cost?", "answer is 69", answer_is_number(69), max_tokens=500),
]

# Category 7: HumanEval-Style (5 questions)
QUESTIONS += [
    Question("HumanEval-Style", "Write a Python function called 'binary_search' that takes a sorted list and a target value, and returns the index of the target if found, or -1 if not found. Use the binary search algorithm (not linear search).", "contains def binary_search, mid-point calculation, and comparison logic", lambda t: "def binary_search" in t and ("mid" in t.lower() or "middle" in t.lower()) and ("//" in t or ">> 1" in t or "/ 2" in t or "mid" in t.lower()), max_tokens=500),
    Question("HumanEval-Style", "Write a Python function called 'merge_sorted' that takes two sorted lists and returns a single merged sorted list. Do NOT use the built-in sort() or sorted() functions — implement the merge step manually.", "contains def merge_sorted with index-based merge logic", lambda t: "def merge_sorted" in t and ("while" in t or "for" in t) and ("append" in t or "extend" in t or "+=" in t or "result" in t.lower()), max_tokens=500),
    Question("HumanEval-Style", "Write a Python function called 'is_prime' that takes an integer n and returns True if n is a prime number and False otherwise. Handle edge cases (n <= 1 should return False).", "contains def is_prime with modulo check and loop", lambda t: "def is_prime" in t and "%" in t and ("range" in t or "while" in t) and ("False" in t or "false" in t.lower()), max_tokens=500),
    Question("HumanEval-Style", "Write a Python class called 'LRUCache' that implements a Least Recently Used cache with a given capacity. It should support 'get(key)' which returns the value or -1 if not found, and 'put(key, value)' which inserts or updates the key-value pair, evicting the least recently used item if the cache is at capacity.", "contains class LRUCache with get/put methods and eviction logic", lambda t: "class LRUCache" in t and ("def get" in t) and ("def put" in t) and ("dict" in t.lower() or "ordereddict" in t.lower() or "hash" in t.lower() or "{}" in t), max_tokens=500),
    Question("HumanEval-Style", "Write a Python function called 'longest_common_prefix' that takes a list of strings and returns the longest common prefix string. If there is no common prefix, return an empty string.", "contains def longest_common_prefix with character comparison logic", lambda t: "def longest_common_prefix" in t and ("for" in t or "while" in t) and ("[" in t) and ('""' in t or "''" in t or "prefix" in t.lower()), max_tokens=500),
]


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def call_chat_completions(host: str, model: str, prompt: str,
                          max_tokens: int = 200, temperature: float = 0.0
                          ) -> tuple:
    """Send one request to /v1/chat/completions and return (text, latency_seconds)."""
    url = f"{host.rstrip('/')}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a helpful assistant. Be concise and precise. "
                "Follow formatting instructions exactly."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
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
        return f"[ERROR: {e}]", time.perf_counter() - t0
    except Exception as e:
        return f"[ERROR: {e}]", time.perf_counter() - t0
    latency = time.perf_counter() - t0

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
        print(f"  [{i}/{total}] {q.category}: {q.prompt[:60]}...", flush=True)

        text, latency = call_chat_completions(host, model, q.prompt,
                                              max_tokens=q.max_tokens)
        try:
            passed = q.check(text)
        except Exception:
            passed = False

        results.append(Result(
            category=q.category,
            prompt=q.prompt,
            expected_desc=q.expected_desc,
            response_text=text,
            passed=passed,
            latency_s=latency,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting & JSON output
# ---------------------------------------------------------------------------

def print_report(results: List[Result], model: str) -> dict:
    """Print per-question details and summary table. Returns summary dict for JSON."""

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"QUALITY BENCHMARK RESULTS — {model} (FP8 KV Cache)")
    print(sep)

    # Per-question detail
    print("\n--- Detailed Results ---\n")
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        truncated = r.response_text.replace("\n", " ")[:120]
        print(f"Q{i:>2} [{status}] ({r.latency_s:.2f}s) [{r.category}]")
        print(f"     Prompt:   {r.prompt[:100]}")
        print(f"     Expected: {r.expected_desc}")
        print(f"     Got:      {truncated}")
        print()

    # Summary table
    categories: dict = {}
    for r in results:
        cat = r.category
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
    category_summary = {}

    for cat in sorted(categories.keys()):
        stats = categories[cat]
        p, t = stats["pass"], stats["total"]
        acc = (p / t * 100) if t else 0
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        print(f"{cat:<24} {p}/{t:<10} {acc:>9.1f}% {avg_lat:>11.2f}s")
        overall_pass += p
        overall_total += t
        category_summary[cat] = {"pass": p, "total": t, "accuracy_pct": round(acc, 1), "avg_latency_s": round(avg_lat, 2)}

    print("-" * len(header))
    overall_acc = (overall_pass / overall_total * 100) if overall_total else 0
    overall_avg = sum(r.latency_s for r in results) / len(results) if results else 0
    print(f"{'OVERALL':<24} {overall_pass}/{overall_total:<10} "
          f"{overall_acc:>9.1f}% {overall_avg:>11.2f}s")

    return {
        "per_category": category_summary,
        "overall_pass": overall_pass,
        "overall_total": overall_total,
        "overall_accuracy_pct": round(overall_acc, 1),
        "overall_avg_latency_s": round(overall_avg, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test 29C: FP8 KV Cache — Accuracy Preservation",
    )
    parser.add_argument("--host", type=str, required=True, help="vLLM server base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                        help="Model name as registered in vLLM")
    args = parser.parse_args()

    print(f"\nTest 29C: FP8 KV Cache — Accuracy Preservation")
    print(f"Target: {args.host} | Model: {args.model}")
    print(f"KV Cache: fp8_e5m2 (compare against Test 7 fp16 baseline)")
    print(f"Questions: {len(QUESTIONS)} across "
          f"{len(set(q.category for q in QUESTIONS))} categories\n")

    results = run_benchmark(args.host, args.model)
    summary = print_report(results, args.model)

    # Save detailed JSON with per-question response text for diff analysis
    json_results = {
        "test": "29C",
        "title": "FP8 KV Cache — Accuracy Preservation",
        "model": args.model,
        "kv_cache_dtype": "fp8_e5m2",
        "num_questions": len(QUESTIONS),
        "summary": summary,
        "per_question": [
            {
                "index": i + 1,
                "category": r.category,
                "prompt": r.prompt,
                "expected_desc": r.expected_desc,
                "response_text": r.response_text,
                "passed": r.passed,
                "latency_s": round(r.latency_s, 3),
            }
            for i, r in enumerate(results)
        ],
    }

    output_file = "stage2_test29c_results.json"
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    # Exit code for CI compatibility
    accuracy = summary["overall_accuracy_pct"]
    if accuracy < 70:
        print(f"WARNING: Overall accuracy {accuracy:.1f}% is below 70% threshold.")
        sys.exit(1)
    else:
        print(f"Accuracy {accuracy:.1f}% meets the 70% threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
