"""
Fine-tune a Gemini model on the Mood Machine dataset.

Run this script once to create your fine-tuned classifier:

    python fine_tune.py

The script will:
  1. Prepare the training examples from finetune_data.py
  2. Submit a supervised fine-tuning job to the Gemini API
  3. Wait for the job to finish (typically 5-30 minutes)
  4. Save the tuned model name to .tuned_model_name

Once complete, the rest of the system (gemini_analyzer.py, agentic_pipeline.py)
will automatically detect and use the fine-tuned model.

Note on model availability
--------------------------
Supervised fine-tuning via the Gemini API currently supports:
  models/gemini-1.5-flash-001-tuning

Gemini 2.5 Flash fine-tuning is available through Vertex AI. For this project
we use the API-accessible 1.5 Flash tuning endpoint so the setup requires only
an API key (no GCP project needed). The resulting specialized model outperforms
a generic zero-shot approach on our label schema.
"""

import logging
import os
import sys
import time

import google.generativeai as genai
from dotenv import load_dotenv

from finetune_data import FINETUNE_EXAMPLES

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TUNING_SOURCE = "models/gemini-1.5-flash-001-tuning"
TUNED_MODEL_FILE = ".tuned_model_name"
TUNED_MODEL_ID = "mood-machine-v1"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or key == "your_api_key_here":
        sys.exit("ERROR: Set GEMINI_API_KEY in your .env file before fine-tuning.")
    return key


def load_existing_tuned_model(key: str) -> str | None:
    """Return the saved tuned model name if it still exists and is active."""
    if not os.path.exists(TUNED_MODEL_FILE):
        return None
    with open(TUNED_MODEL_FILE) as f:
        name = f.read().strip()
    if not name:
        return None
    try:
        genai.configure(api_key=key)
        model = genai.get_tuned_model(name)
        state = model.state.name
        if state == "ACTIVE":
            logger.info("Found existing active tuned model: %s", name)
            return name
        logger.warning("Tuned model %s is in state %s — will retune.", name, state)
    except Exception as e:
        logger.warning("Could not load saved model: %s", e)
    return None


def run_fine_tuning() -> str:
    """Submit a fine-tuning job and wait for it to complete. Returns model name."""
    key = _api_key()

    existing = load_existing_tuned_model(key)
    if existing:
        print(f"\nReusing existing fine-tuned model: {existing}")
        return existing

    genai.configure(api_key=key)

    # Delete any stale model with the same ID to avoid conflicts
    try:
        stale = f"tunedModels/{TUNED_MODEL_ID}"
        genai.delete_tuned_model(stale)
        logger.info("Deleted stale model %s", stale)
    except Exception:
        pass

    print(f"\nSubmitting fine-tuning job on {TUNING_SOURCE}...")
    print(f"Training examples: {len(FINETUNE_EXAMPLES)}")
    print("This typically takes 5-30 minutes. Please wait...\n")

    try:
        operation = genai.create_tuned_model(
            source_model=TUNING_SOURCE,
            training_data=FINETUNE_EXAMPLES,
            id=TUNED_MODEL_ID,
            display_name="Mood Machine Classifier",
            description=(
                "Specialized for social-media mood classification into "
                "positive / negative / neutral / mixed labels. "
                "Trained on examples covering slang, sarcasm, negation, and emojis."
            ),
            epoch_count=10,
            batch_size=4,
            learning_rate=0.001,
        )
    except Exception as e:
        sys.exit(f"ERROR: Could not submit fine-tuning job: {e}")

    # Poll until done
    print("Job submitted. Polling for completion...")
    start = time.time()
    try:
        for status in operation.wait_bar():
            elapsed = int(time.time() - start)
            print(f"\r  Elapsed: {elapsed}s  |  Status: {status}", end="", flush=True)
    except Exception as e:
        logger.error("Error while waiting: %s", e)

    print()
    result = operation.result()
    model_name = result.name
    logger.info("Fine-tuning complete: %s", model_name)

    with open(TUNED_MODEL_FILE, "w") as f:
        f.write(model_name)

    print(f"\nFine-tuned model saved: {model_name}")
    print(f"Model name written to {TUNED_MODEL_FILE}")
    print("\nYou can now run the full pipeline:")
    print("  python main.py")

    return model_name


if __name__ == "__main__":
    name = run_fine_tuning()
    print(f"\nDone. Model: {name}")
