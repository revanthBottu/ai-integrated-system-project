"""
Mood Machine — AI Integrated System
Entry point with four operating modes:

  1  Quick      Rule-based only (offline, no API key needed)
  2  Deep AI    Full agentic pipeline: fine-tuned/specialized Gemini 2.5 Flash
                + ML cross-validation + step trace
  3  Compare    Side-by-side predictions from all three models
  4  Test       Run the full reliability test suite

Usage:
    python main.py            # prompts for mode
    python main.py --mode 2   # jump straight to Deep AI
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING)


# ── Mode 1: Quick (offline) ───────────────────────────────────────────────────

def run_quick_mode() -> None:
    from dataset import SAMPLE_POSTS, TRUE_LABELS
    from mood_analyzer import MoodAnalyzer

    analyzer = MoodAnalyzer()
    correct = 0

    print("\n=== Quick Mode: Rule-Based Analysis ===")
    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        predicted = analyzer.predict_label(text)
        explain = analyzer.explain(text)
        correct += predicted == true_label
        tick = "✓" if predicted == true_label else "✗"
        print(f'{tick} "{text}"')
        print(f'   predicted={predicted}  true={true_label}  |  {explain}\n')

    print(f"Rule-based accuracy: {correct / len(SAMPLE_POSTS):.0%}\n")

    print("=== Interactive (type 'quit' to exit) ===")
    while True:
        text = input("You: ").strip()
        if not text or text.lower() == "quit":
            break
        label = analyzer.predict_label(text)
        explain = analyzer.explain(text)
        print(f"Mood: {label}  |  {explain}\n")


# ── Mode 2: Deep AI ───────────────────────────────────────────────────────────

def run_deep_mode() -> None:
    from agentic_pipeline import AgenticMoodPipeline

    print("\nInitializing AI pipeline...")
    pipeline = AgenticMoodPipeline()

    print(f"\n=== Deep AI Mode: Agentic Pipeline ===")
    print(f"Gemini model: {pipeline._gemini.model_description}")
    print("Commands: 'verbose' toggles step trace | 'quit' exits\n")

    verbose = False
    while True:
        text = input("You: ").strip()
        if not text or text.lower() == "quit":
            break
        if text.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}\n")
            continue

        result = pipeline.analyze(text)
        print(pipeline.format_result(result, verbose=verbose))
        print()


# ── Mode 3: Compare ───────────────────────────────────────────────────────────

def run_compare_mode() -> None:
    from agentic_pipeline import AgenticMoodPipeline
    from dataset import SAMPLE_POSTS, TRUE_LABELS

    print("\nInitializing all models...")
    pipeline = AgenticMoodPipeline()

    header = f"{'Text':<42} {'Rule':>10} {'ML':>10} {'Gemini':>10} {'True':>10}"
    print(f"\n=== Compare Mode: All Three Models ===\n{header}")
    print("─" * 82)

    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        result = pipeline.analyze(text)
        display = text[:39] + "..." if len(text) > 42 else text
        agree = "✓" if result.models_agreed else " "
        print(
            f"{agree} {display:<40} "
            f"{result.rule_based_label:>10} "
            f"{result.ml_label:>10} "
            f"{result.gemini_label:>10} "
            f"{true_label:>10}"
        )

    print("\n=== Interactive compare (type 'quit' to exit) ===")
    while True:
        text = input("You: ").strip()
        if not text or text.lower() == "quit":
            break
        result = pipeline.analyze(text)
        print(f"  Rule-based : {result.rule_based_label}")
        print(f"  ML model   : {result.ml_label}")
        print(f"  Gemini AI  : {result.gemini_label}  (conf={result.gemini_confidence:.0%})")
        print(f"  Final      : {result.final_label.upper()}\n")


# ── Mode 4: Test ──────────────────────────────────────────────────────────────

def run_test_mode() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)
    from agentic_pipeline import AgenticMoodPipeline
    from reliability_tests import run_all_tests

    print("\nInitializing pipeline for reliability testing...")
    pipeline = AgenticMoodPipeline()
    run_all_tests(pipeline)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mood Machine — AI Integrated System")
    parser.add_argument(
        "--mode",
        choices=["1", "2", "3", "4"],
        help="1=Quick, 2=Deep AI, 3=Compare, 4=Test",
    )
    args = parser.parse_args()

    if not args.mode:
        print("=" * 52)
        print("     MOOD MACHINE — AI Integrated System")
        print("=" * 52)
        print("\n  1  Quick    — Rule-based (offline)")
        print("  2  Deep AI  — Gemini fine-tuned + agentic pipeline")
        print("  3  Compare  — Side-by-side: all three models")
        print("  4  Test     — Reliability test suite\n")
        args.mode = input("Choose a mode (1-4): ").strip()

    dispatch = {"1": run_quick_mode, "2": run_deep_mode,
                "3": run_compare_mode, "4": run_test_mode}

    fn = dispatch.get(args.mode)
    if not fn:
        print("Invalid choice. Use 1, 2, 3, or 4.")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
