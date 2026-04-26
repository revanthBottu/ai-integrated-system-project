"""
Agentic mood analysis pipeline.

Coordinates three complementary classifiers in a transparent, logged workflow:

  Step 1 — Input guardrail      (length / empty / encoding checks)
  Step 2 — Rule-based analysis  (mood_analyzer.MoodAnalyzer)
  Step 3 — Gemini analysis      (fine-tuned model or specialized Gemini 2.5 Flash)
  Step 4 — Cross-validation     (ML model as tiebreaker on disagreement)
  Step 5 — Synthesis            (final label + confidence + full trace)

Why three models? Each catches different failure modes:
  - Rule-based: transparent, fast, good at obvious cases
  - Gemini (fine-tuned/specialized): understands sarcasm, slang, context
  - ML (logistic regression): learned patterns from our labeled examples
Cross-validation means no single model can silently be wrong.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from dataset import SAMPLE_POSTS, TRUE_LABELS
from gemini_analyzer import GeminiAnalysis, GeminiMoodAnalyzer
from ml_experiments import predict_single_text, train_ml_model
from mood_analyzer import MoodAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MAX_LEN = 500


@dataclass
class PipelineResult:
    text: str
    final_label: str
    final_confidence: float
    rule_based_label: str
    ml_label: str
    gemini_label: str
    gemini_confidence: float
    gemini_reasoning: str
    models_agreed: bool
    tiebreaker_used: bool
    sarcasm_detected: bool
    negation_detected: bool
    slang_detected: bool
    used_finetuned: bool
    steps_log: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


def _empty_analysis(label: str, reason: str) -> GeminiAnalysis:
    return GeminiAnalysis(
        label=label,
        confidence=0.5,
        reasoning=reason,
        sarcasm_detected=False,
        negation_detected=False,
        slang_detected=False,
        emojis_detected=False,
        used_finetuned=False,
        raw_response="",
    )


class AgenticMoodPipeline:
    """
    End-to-end agentic pipeline for mood analysis.

    Initializing this class trains the ML model (once), connects to Gemini,
    and loads the fine-tuned model if available. All three models are kept
    alive for the duration of the session.
    """

    def __init__(self) -> None:
        logger.info("Initializing AgenticMoodPipeline...")
        self._rule = MoodAnalyzer()
        self._vectorizer, self._ml = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
        self._gemini = GeminiMoodAnalyzer()
        logger.info(
            "Pipeline ready — Gemini model: %s", self._gemini.model_description
        )

    # ──────────────────────────────────────────────────────────────────

    def _validate(self, text: str) -> Optional[str]:
        stripped = text.strip() if text else ""
        if not stripped:
            return "Input is empty."
        if len(text) > _MAX_LEN:
            return f"Input exceeds {_MAX_LEN} characters (got {len(text)})."
        return None

    # ──────────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> PipelineResult:
        """
        Run the full 5-step agentic pipeline on a text input.
        Returns a PipelineResult with the final label, confidence, per-model
        predictions, detected features, and a full step trace for auditability.
        """
        t0 = time.time()
        steps: List[str] = []

        # ── Step 1: Guardrail ──────────────────────────────────────────
        steps.append("Step 1: Input guardrail")
        err = self._validate(text)
        if err:
            steps.append(f"  → BLOCKED: {err}")
            logger.warning("Guardrail blocked: %s", err)
            return PipelineResult(
                text=text, final_label="neutral", final_confidence=0.0,
                rule_based_label="neutral", ml_label="neutral",
                gemini_label="neutral", gemini_confidence=0.0,
                gemini_reasoning=err, models_agreed=True,
                tiebreaker_used=False, sarcasm_detected=False,
                negation_detected=False, slang_detected=False,
                used_finetuned=False, steps_log=steps,
            )
        steps.append(f"  → Valid ({len(text)} chars)")

        # ── Step 2: Rule-based ─────────────────────────────────────────
        steps.append("Step 2: Rule-based analysis")
        rule_label = self._rule.predict_label(text)
        rule_explain = self._rule.explain(text)
        steps.append(f"  → {rule_label}  |  {rule_explain}")
        logger.info("[rule] '%s...' → %s", text[:50], rule_label)

        # ── Step 3: Gemini ─────────────────────────────────────────────
        steps.append("Step 3: Gemini analysis (fine-tuned / specialized)")
        try:
            gem: GeminiAnalysis = self._gemini.analyze(text)
        except Exception as exc:
            steps.append(f"  → Gemini unavailable ({exc}); using rule-based fallback")
            logger.error("Gemini error: %s", exc)
            gem = _empty_analysis(rule_label, f"Gemini unavailable: {exc}")

        gemini_label = gem.label
        model_tag = "fine-tuned" if gem.used_finetuned else "specialized"
        steps.append(f"  → {gemini_label}  |  confidence {gem.confidence:.0%}  [{model_tag}]")
        steps.append(f"  → Reasoning: {gem.reasoning}")
        for flag, name in [
            (gem.sarcasm_detected, "sarcasm"),
            (gem.negation_detected, "negation"),
            (gem.slang_detected, "slang"),
        ]:
            if flag:
                steps.append(f"  → {name} flag raised")
        logger.info(
            "[gemini/%s] '%s...' → %s (%.0f%%)",
            model_tag, text[:50], gemini_label, gem.confidence * 100,
        )

        # ── Step 4: Cross-validation ───────────────────────────────────
        steps.append("Step 4: Cross-validation")
        tiebreaker_used = False
        ml_label = "neutral"

        # Gemini with high confidence and "mixed" — trust it (rule-based can't produce this)
        if gemini_label == "mixed" and gem.confidence >= 0.65:
            final_label = "mixed"
            final_confidence = gem.confidence
            models_agreed = False
            steps.append(f"  → High-confidence 'mixed' from Gemini accepted (rule={rule_label})")

        elif rule_label == gemini_label:
            final_label = gemini_label
            final_confidence = min(1.0, gem.confidence + 0.10)
            models_agreed = True
            steps.append(f"  → AGREE on '{final_label}' — confidence boosted to {final_confidence:.0%}")

        else:
            steps.append(f"  → DISAGREE: rule='{rule_label}' gemini='{gemini_label}'")
            ml_label = predict_single_text(text, self._vectorizer, self._ml)
            tiebreaker_used = True
            models_agreed = False
            steps.append(f"  → ML tiebreaker: {ml_label}")
            logger.info("[ml] '%s...' → %s", text[:50], ml_label)

            if ml_label == gemini_label:
                final_label = gemini_label
                final_confidence = gem.confidence
                steps.append(f"  → ML confirms Gemini → '{final_label}'")
            elif ml_label == rule_label:
                final_label = rule_label
                final_confidence = 0.60
                steps.append(f"  → ML confirms rule-based → '{final_label}'")
            else:
                # All three differ — defer to Gemini (most expressive model)
                final_label = gemini_label
                final_confidence = max(0.40, gem.confidence - 0.10)
                steps.append(f"  → All three disagree; defer to Gemini → '{final_label}'")

        # ── Step 5: Synthesis ──────────────────────────────────────────
        steps.append("Step 5: Synthesis")
        steps.append(f"  → Final: {final_label}  |  confidence: {final_confidence:.0%}")

        elapsed = (time.time() - t0) * 1000
        logger.info("Done: '%s...' → %s  (%.0f ms)", text[:50], final_label, elapsed)

        return PipelineResult(
            text=text,
            final_label=final_label,
            final_confidence=final_confidence,
            rule_based_label=rule_label,
            ml_label=ml_label,
            gemini_label=gemini_label,
            gemini_confidence=gem.confidence,
            gemini_reasoning=gem.reasoning,
            models_agreed=models_agreed,
            tiebreaker_used=tiebreaker_used,
            sarcasm_detected=gem.sarcasm_detected,
            negation_detected=gem.negation_detected,
            slang_detected=gem.slang_detected,
            used_finetuned=gem.used_finetuned,
            steps_log=steps,
            latency_ms=elapsed,
        )

    # ──────────────────────────────────────────────────────────────────

    def format_result(self, result: PipelineResult, verbose: bool = False) -> str:
        model_tag = "[fine-tuned]" if result.used_finetuned else "[specialized]"
        lines = [
            f'\nText:       "{result.text}"',
            f"Mood:       {result.final_label.upper()}  {model_tag}",
            f"Confidence: {result.final_confidence:.0%}",
            f"Reasoning:  {result.gemini_reasoning}",
        ]

        flags = [
            name for name, on in [
                ("sarcasm", result.sarcasm_detected),
                ("negation", result.negation_detected),
                ("slang", result.slang_detected),
            ] if on
        ]
        if flags:
            lines.append(f"Detected:   {', '.join(flags)}")

        if result.tiebreaker_used:
            lines.append(
                f"Note:       Models disagreed "
                f"(rule={result.rule_based_label}, "
                f"gemini={result.gemini_label}, "
                f"ml={result.ml_label})"
            )

        if verbose:
            lines.append("\n─── Pipeline trace ───")
            lines.extend(f"  {s}" for s in result.steps_log)
            lines.append(f"  Latency: {result.latency_ms:.0f} ms")

        return "\n".join(lines)
