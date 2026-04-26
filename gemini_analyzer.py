"""
Gemini-powered mood analyzer.

Loads the fine-tuned classifier from .tuned_model_name if available.
Falls back to Gemini 2.5 Flash with a specialized system instruction that
constrains it to our exact label schema and teaches it our dataset's quirks
(slang, sarcasm, emojis, negation).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TUNED_MODEL_FILE = ".tuned_model_name"
BASE_MODEL = "gemini-2.5-flash-preview-05-20"
VALID_LABELS = {"positive", "negative", "neutral", "mixed"}

_SYSTEM_INSTRUCTION = """\
You are a specialized mood classifier for short social media posts and messages.

Your ONLY job is to classify each text into exactly one of these four labels:
  positive  — happiness, excitement, joy, pride, enthusiasm
  negative  — sadness, anger, frustration, disappointment, sarcasm, despair
  neutral   — factual, ambiguous, or emotionally flat
  mixed     — simultaneous positive AND negative signals (e.g., "tired but hopeful")

Special rules you MUST follow:
1. Sarcasm: positive words in a negative context → label negative
   (e.g., "Love getting stuck in traffic 🙄" → negative)
2. Negation: "not happy" → negative; "not bad" → positive
3. Slang: "cooked" (failing) → negative; "cooking" (doing great) → positive;
   "buns" → negative; "W" → positive; "L" → negative; "mid" → negative
4. Emojis: 😭💀😤👎 → negative signal; 🎉🔥😊❤️👍 → positive signal
5. "i'm fine. totally fine." (repeated reassurance) → negative (passive distress)
6. "mixed" is for texts that contain BOTH clear positive and negative emotions.

You must respond ONLY with valid JSON, no markdown, no extra text:
{
  "label": "<positive|negative|neutral|mixed>",
  "confidence": <0.0–1.0>,
  "reasoning": "<one sentence>",
  "features": {
    "sarcasm": <true|false>,
    "negation": <true|false>,
    "slang": <true|false>,
    "emojis": <true|false>
  }
}
"""

_CLASSIFY_PROMPT = 'Classify this text: "{text}"'


@dataclass
class GeminiAnalysis:
    label: str
    confidence: float
    reasoning: str
    sarcasm_detected: bool
    negation_detected: bool
    slang_detected: bool
    emojis_detected: bool
    used_finetuned: bool
    raw_response: str


class GeminiMoodAnalyzer:
    """
    Mood analyzer backed by a fine-tuned or specialized Gemini model.

    On first use the class checks for a fine-tuned model saved by fine_tune.py.
    If found, label prediction comes from the fine-tuned model. Otherwise, the
    base Gemini 2.5 Flash model is used with a specialized system instruction.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key or key == "your_api_key_here":
            raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")
        self._client = genai.Client(api_key=key)
        self._tuned_name = self._load_tuned_model_name()
        if self._tuned_name:
            logger.info("Using fine-tuned model: %s", self._tuned_name)
        else:
            logger.info("No fine-tuned model found — using specialized base model")

    def _load_tuned_model_name(self) -> Optional[str]:
        p = Path(TUNED_MODEL_FILE)
        if not p.exists():
            return None
        name = p.read_text().strip()
        if not name:
            return None
        try:
            self._client.models.get(model=name)
            return name
        except Exception as e:
            logger.warning("Tuned model %s not accessible: %s — falling back to base", name, e)
        return None

    def _classify_with_finetuned(self, text: str) -> Optional[str]:
        if not self._tuned_name:
            return None
        try:
            response = self._client.models.generate_content(
                model=self._tuned_name,
                contents=text,
                config=types.GenerateContentConfig(temperature=0),
            )
            label = response.text.strip().lower().strip(".,!? ")
            return label if label in VALID_LABELS else None
        except Exception as e:
            logger.error("Fine-tuned model error: %s", e)
            return None

    def _analyze_with_base(self, text: str) -> GeminiAnalysis:
        prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', "'"))
        raw = ""
        try:
            response = self._client.models.generate_content(
                model=BASE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            data = json.loads(raw)
            label = str(data.get("label", "neutral")).lower()
            if label not in VALID_LABELS:
                label = "neutral"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            features = data.get("features", {})

            return GeminiAnalysis(
                label=label,
                confidence=confidence,
                reasoning=str(data.get("reasoning", "No reasoning provided.")),
                sarcasm_detected=bool(features.get("sarcasm", False)),
                negation_detected=bool(features.get("negation", False)),
                slang_detected=bool(features.get("slang", False)),
                emojis_detected=bool(features.get("emojis", False)),
                used_finetuned=False,
                raw_response=raw,
            )
        except json.JSONDecodeError as e:
            logger.error("JSON parse failed: %s | raw=%s", e, raw[:200])
            return GeminiAnalysis(
                label="neutral",
                confidence=0.0,
                reasoning="Could not parse AI response.",
                sarcasm_detected=False,
                negation_detected=False,
                slang_detected=False,
                emojis_detected=False,
                used_finetuned=False,
                raw_response=raw,
            )

    def analyze(self, text: str) -> GeminiAnalysis:
        """Classify mood using the best available Gemini model."""
        if not text or not text.strip():
            return GeminiAnalysis(
                label="neutral",
                confidence=1.0,
                reasoning="Empty input has no sentiment.",
                sarcasm_detected=False,
                negation_detected=False,
                slang_detected=False,
                emojis_detected=False,
                used_finetuned=False,
                raw_response="",
            )

        result = self._analyze_with_base(text)
        logger.debug("[base] '%s...' → %s (%.0f%%)", text[:40], result.label, result.confidence * 100)

        ft_label = self._classify_with_finetuned(text)
        if ft_label:
            logger.debug("[finetuned] '%s...' → %s", text[:40], ft_label)
            if ft_label != result.label:
                logger.info(
                    "Fine-tuned overrides base: %s → %s (text: '%s...')",
                    result.label, ft_label, text[:40],
                )
                result.confidence = max(0.45, result.confidence - 0.10)
            result.label = ft_label
            result.used_finetuned = True

        return result

    @property
    def model_description(self) -> str:
        if self._tuned_name:
            return f"Fine-tuned ({self._tuned_name})"
        return f"Specialized base model ({BASE_MODEL})"
