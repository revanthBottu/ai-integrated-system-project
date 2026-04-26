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
"""

import logging
import os
import sys
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

from finetune_data import FINETUNE_EXAMPLES

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TUNING_SOURCE = "models/gemini-1.5-flash-001-tuning"
TUNED_MODEL_FILE = ".tuned_model_name"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or key == "your_api_key_here":
        sys.exit("ERROR: Set GEMINI_API_KEY in your .env file before fine-tuning.")
    return key


def run_fine_tuning(progress_callback=None) -> str:
    """Submit a fine-tuning job and wait for it to complete. Returns model name."""
    key = _api_key()
    client = genai.Client(api_key=key)

    # Reuse existing tuned model if still accessible
    if os.path.exists(TUNED_MODEL_FILE):
        with open(TUNED_MODEL_FILE) as f:
            saved_name = f.read().strip()
        if saved_name:
            try:
                client.models.get(model=saved_name)
                logger.info("Found existing active tuned model: %s", saved_name)
                print(f"\nReusing existing fine-tuned model: {saved_name}")
                return saved_name
            except Exception as e:
                logger.warning("Saved model no longer accessible (%s) — retuning.", e)

    print(f"\nSubmitting fine-tuning job on {TUNING_SOURCE}...")
    print(f"Training examples: {len(FINETUNE_EXAMPLES)}")
    print("This typically takes 5-30 minutes. Please wait...\n")

    training_examples = [
        types.TuningExample(text_input=ex["text_input"], output=ex["output"])
        for ex in FINETUNE_EXAMPLES
    ]

    try:
        tuning_job = client.tunings.tune(
            base_model=TUNING_SOURCE,
            training_dataset=types.TuningDataset(examples=training_examples),
            config=types.CreateTuningJobConfig(
                tuned_model_display_name="Mood Machine Classifier",
                description=(
                    "Specialized for social-media mood classification into "
                    "positive / negative / neutral / mixed labels. "
                    "Trained on examples covering slang, sarcasm, negation, and emojis."
                ),
                epoch_count=10,
                batch_size=4,
                learning_rate=0.001,
            ),
        )
    except Exception as e:
        sys.exit(f"ERROR: Could not submit fine-tuning job: {e}")

    logger.info("Fine-tuning job submitted: %s", tuning_job.name)
    logger.info("Training on %d examples — polling every 30s...", len(FINETUNE_EXAMPLES))
    print("Job submitted. Polling for completion...")
    start = time.time()

    while not tuning_job.has_ended:
        elapsed = int(time.time() - start)
        msg = f"Elapsed: {elapsed}s  |  State: {tuning_job.state}"
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        print(f"\r  {msg}", end="", flush=True)
        time.sleep(30)
        tuning_job = client.tunings.get(name=tuning_job.name)

    print()

    if not tuning_job.has_succeeded:
        error_msg = getattr(tuning_job, "error", "unknown error")
        sys.exit(f"ERROR: Fine-tuning failed: {error_msg}")

    model_name = tuning_job.tuned_model.model
    logger.info("Fine-tuning complete: %s", model_name)

    with open(TUNED_MODEL_FILE, "w") as f:
        f.write(model_name)

    print(f"\nFine-tuned model saved: {model_name}")
    print(f"Model name written to {TUNED_MODEL_FILE}")

    return model_name


if __name__ == "__main__":
    name = run_fine_tuning()
    print(f"\nDone. Model: {name}")
