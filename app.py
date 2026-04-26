"""
Flask web server for Mood Machine.

Startup sequence:
  1. GeminiMoodAnalyzer initializes — loads fine-tuned model if .tuned_model_name exists.
  2. If no fine-tuned model, a background fine-tuning job is submitted to the Gemini API
     (takes 5–30 min). The base model handles requests while fine-tuning runs.
  3. When fine-tuning completes, the analyzer is hot-swapped to use the new model.

Endpoints:
  GET  /        — frontend HTML
  GET  /status  — JSON: pipeline state, finetune state, model description
  POST /analyze — JSON body {text}: returns label, confidence, reasoning, features
"""

import logging
import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Shared state (thread-safe) ────────────────────────────────────────────────

_analyzer = None
_lock = threading.Lock()

_status = {
    "pipeline": "initializing",   # initializing | ready | error
    "finetune": "checking",        # checking | not_needed | running | done | error
    "model": "unknown",
    "finetune_message": "",
}
_status_lock = threading.Lock()


def _set(**kw):
    with _status_lock:
        _status.update(kw)


def _get_status():
    with _status_lock:
        return dict(_status)


# ── Background workers ────────────────────────────────────────────────────────

def _init_worker():
    """Initialize the Gemini analyzer and optionally kick off fine-tuning."""
    global _analyzer
    try:
        from gemini_analyzer import GeminiMoodAnalyzer
        a = GeminiMoodAnalyzer()
        with _lock:
            _analyzer = a
        _set(pipeline="ready", model=a.model_description)
        logger.info("Analyzer ready — %s", a.model_description)

        if a._tuned_name:
            _set(finetune="not_needed", finetune_message="Fine-tuned model already active.")
        else:
            key = os.environ.get("GEMINI_API_KEY", "")
            if not key or key == "your_api_key_here":
                _set(
                    finetune="error",
                    finetune_message="GEMINI_API_KEY not set — skipping fine-tuning.",
                )
            else:
                _set(finetune="running", finetune_message="Fine-tuning job submitted (5–30 min)…")
                threading.Thread(target=_finetune_worker, daemon=True).start()

    except Exception as exc:
        logger.error("Init failed: %s", exc)
        _set(pipeline="error", finetune_message=str(exc))


def _finetune_worker():
    """Run fine-tuning in the background; hot-swap analyzer when done."""
    global _analyzer
    try:
        from fine_tune import run_fine_tuning
        name = run_fine_tuning()
        _set(finetune="done", finetune_message=f"Ready: {name}")
        logger.info("Fine-tuning complete — %s", name)

        from gemini_analyzer import GeminiMoodAnalyzer
        new_a = GeminiMoodAnalyzer()
        with _lock:
            _analyzer = new_a
        _set(model=new_a.model_description)
        logger.info("Analyzer hot-swapped to fine-tuned model")

    except SystemExit as exc:
        _set(finetune="error", finetune_message=f"Fine-tuning aborted: {exc}")
        logger.error("Fine-tuning sys.exit: %s", exc)
    except Exception as exc:
        _set(finetune="error", finetune_message=str(exc))
        logger.error("Fine-tuning failed: %s", exc)


# Start initialization immediately on import (safe with debug=False / no reloader)
threading.Thread(target=_init_worker, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/status")
def status():
    return jsonify(_get_status())


@app.post("/analyze")
def analyze():
    with _lock:
        a = _analyzer

    if a is None:
        return jsonify({"error": "Still initializing — please wait a moment."}), 503

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400
    if len(text) > 500:
        return jsonify({"error": f"Text too long ({len(text)} chars; max 500)."}), 400

    try:
        result = a.analyze(text)
        features = [
            name
            for name, on in [
                ("sarcasm", result.sarcasm_detected),
                ("negation", result.negation_detected),
                ("slang", result.slang_detected),
                ("emojis", result.emojis_detected),
            ]
            if on
        ]
        return jsonify({
            "label": result.label,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "features": features,
            "used_finetuned": result.used_finetuned,
        })
    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
