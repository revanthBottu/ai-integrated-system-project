"""
Reliability and evaluation test suite for the Mood Machine AI system.

Tests:
  1. Benchmark     — full pipeline accuracy on SAMPLE_POSTS / TRUE_LABELS
  2. Consistency   — same input must produce the same label across 3 runs
  3. Edge cases    — empty, emojis-only, ALL CAPS, gibberish, double negation
  4. Guardrail     — oversized input must be blocked by Step 1
  5. Calibration   — high-confidence predictions should be more accurate

Run:
    python reliability_tests.py
"""

import logging
import time
from typing import List, Tuple

from agentic_pipeline import AgenticMoodPipeline, PipelineResult
from dataset import SAMPLE_POSTS, TRUE_LABELS

logger = logging.getLogger(__name__)


# ── Test 1: Benchmark ─────────────────────────────────────────────────────────

def run_benchmark(pipeline: AgenticMoodPipeline) -> Tuple[float, List[dict]]:
    print("\n" + "=" * 62)
    print("TEST 1: Benchmark — full pipeline vs labeled dataset")
    print("=" * 62)

    records = []
    correct = 0

    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        result: PipelineResult = pipeline.analyze(text)
        ok = result.final_label == true_label
        if ok:
            correct += 1
        records.append(
            dict(
                text=text,
                true=true_label,
                predicted=result.final_label,
                confidence=result.final_confidence,
                correct=ok,
                used_finetuned=result.used_finetuned,
            )
        )
        tick = "✓" if ok else "✗"
        display = f'"{text[:48]}..."' if len(text) > 50 else f'"{text}"'
        print(f"  {tick} {display}")
        print(
            f"     predicted={result.final_label:<10} true={true_label:<10} "
            f"conf={result.final_confidence:.0%}  "
            f"{'[fine-tuned]' if result.used_finetuned else '[specialized]'}"
        )

    accuracy = correct / len(SAMPLE_POSTS)
    print(f"\n  Benchmark accuracy: {accuracy:.0%}  ({correct}/{len(SAMPLE_POSTS)})")
    return accuracy, records


# ── Test 2: Consistency ───────────────────────────────────────────────────────

def run_consistency_test(pipeline: AgenticMoodPipeline, runs: int = 3) -> bool:
    print("\n" + "=" * 62)
    print(f"TEST 2: Consistency — same input across {runs} runs")
    print("=" * 62)

    probes = [
        "I absolutely love this!",
        "Today was the worst day of my life.",
        "I absolutely love getting stuck in traffic",
        "Feeling tired but kind of hopeful",
    ]

    all_consistent = True
    for text in probes:
        labels = [pipeline.analyze(text).final_label for _ in range(runs)]
        consistent = len(set(labels)) == 1
        if not consistent:
            all_consistent = False
        tick = "✓" if consistent else "✗"
        note = "consistent" if consistent else "INCONSISTENT"
        print(f"  {tick} \"{text}\"")
        print(f"     {labels}  →  {note}")

    return all_consistent


# ── Test 3: Edge cases ────────────────────────────────────────────────────────

def run_edge_case_tests(pipeline: AgenticMoodPipeline) -> int:
    print("\n" + "=" * 62)
    print("TEST 3: Edge cases")
    print("=" * 62)

    cases = [
        ("", "neutral", "empty string → guardrail"),
        ("😊😊😊", "positive", "positive emojis only"),
        ("😭😭😭", "negative", "negative emojis only"),
        ("TERRIBLE AWFUL HORRIBLE", "negative", "all-caps negative"),
        ("AMAZING WONDERFUL FANTASTIC", "positive", "all-caps positive"),
        ("I am not unhappy about this", "positive", "double negation"),
        ("abc123 xyz987", "neutral", "gibberish / no sentiment"),
        ("lol", "neutral", "single ambiguous token"),
    ]

    passed = 0
    for text, expected, description in cases:
        result = pipeline.analyze(text)
        ok = result.final_label == expected
        if ok:
            passed += 1
        tick = "✓" if ok else "?"
        label_info = f"predicted={result.final_label}"
        if not ok:
            label_info += f"  (expected={expected})"
        print(f"  {tick} {description}")
        print(f"     {label_info}")

    total = len(cases)
    print(f"\n  Edge cases: {passed}/{total} matched expected")
    return passed


# ── Test 4: Guardrail ─────────────────────────────────────────────────────────

def run_guardrail_test(pipeline: AgenticMoodPipeline) -> bool:
    print("\n" + "=" * 62)
    print("TEST 4: Guardrail — oversized input")
    print("=" * 62)

    oversized = "This is a very long input. " * 40  # ~1 000 chars
    result = pipeline.analyze(oversized)

    blocked = result.final_confidence == 0.0 and "exceeds" in result.gemini_reasoning
    tick = "✓" if blocked else "✗"
    print(f"  {tick} Oversized input ({len(oversized)} chars)")
    print(f"     blocked={blocked}  |  reason: {result.gemini_reasoning[:80]}")
    return blocked


# ── Test 5: Confidence calibration ────────────────────────────────────────────

def run_calibration_test(pipeline: AgenticMoodPipeline) -> None:
    print("\n" + "=" * 62)
    print("TEST 5: Confidence calibration")
    print("=" * 62)

    threshold = 0.75
    buckets: dict = {"high": [0, 0], "low": [0, 0]}  # [correct, total]

    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        result = pipeline.analyze(text)
        correct = int(result.final_label == true_label)
        if result.final_confidence >= threshold:
            buckets["high"][0] += correct
            buckets["high"][1] += 1
        else:
            buckets["low"][0] += correct
            buckets["low"][1] += 1

    for bucket, (correct, total) in buckets.items():
        if total:
            acc = correct / total
            label = f">= {threshold:.0%}" if bucket == "high" else f"<  {threshold:.0%}"
            print(f"  Confidence {label}: {acc:.0%} accuracy ({correct}/{total})")

    high_acc = buckets["high"][0] / buckets["high"][1] if buckets["high"][1] else 0
    low_acc = buckets["low"][0] / buckets["low"][1] if buckets["low"][1] else 0
    if buckets["high"][1] and buckets["low"][1]:
        if high_acc >= low_acc:
            print("  ✓ Higher confidence correlates with higher accuracy")
        else:
            print("  ? Low-confidence predictions were more accurate — check calibration")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all_tests(pipeline: AgenticMoodPipeline) -> None:
    print("\n" + "=" * 62)
    print("  MOOD MACHINE — Reliability Test Suite")
    print("=" * 62)
    t0 = time.time()

    acc, _ = run_benchmark(pipeline)
    consistent = run_consistency_test(pipeline)
    edge_passed = run_edge_case_tests(pipeline)
    guardrail_ok = run_guardrail_test(pipeline)
    run_calibration_test(pipeline)

    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Benchmark accuracy : {acc:.0%}")
    print(f"  Consistency        : {'PASS ✓' if consistent else 'FAIL ✗'}")
    print(f"  Edge cases matched : {edge_passed}/8")
    print(f"  Guardrail          : {'PASS ✓' if guardrail_ok else 'FAIL ✗'}")
    print(f"  Total test time    : {elapsed:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("Initializing pipeline...")
    pl = AgenticMoodPipeline()
    run_all_tests(pl)
