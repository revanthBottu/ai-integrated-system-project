# Mood Machine — AI Integrated System

A multi-model mood classifier that combines a fine-tuned Gemini classifier,
specialized Gemini 2.5 Flash, a rule-based analyzer, and a logistic regression
model in a transparent agentic pipeline with built-in reliability testing.

> **Loom walkthrough:** [_\[Add your Loom link here after recording\]_](https://www.loom.com/share/7c3a595aba2040a9ba214291241aaa1f)

---

## Original Project

**Base project:** Mood Machine (AI110 Modules 1–3)

The original Mood Machine classified short social-media posts into mood labels
(positive / negative / neutral / mixed) using two approaches: a hand-crafted
rule-based scorer (`mood_analyzer.py`) that assigned weights to positive and
negative words, and a scikit-learn logistic regression model (`ml_experiments.py`)
trained on a small labeled dataset. Both ran entirely offline with no external API.

---

## What This System Does

The expanded system keeps both original classifiers and adds a third, more powerful
signal: a **fine-tuned Gemini model** (or Gemini 2.5 Flash with a specialized
system instruction when the fine-tuned model is not yet available). An **agentic
pipeline** coordinates all three, cross-validates their predictions, and uses the
ML model as a transparent tiebreaker when they disagree. Every decision step is
logged. A separate **reliability test suite** measures accuracy, consistency, edge
cases, and confidence calibration.

---

## Architecture Overview

```
User Input
  → Step 1: Input guardrail          (blocks empty / oversized inputs)
  → Step 2: Rule-based analysis      (mood_analyzer.MoodAnalyzer)
  → Step 3: Gemini analysis          (fine-tuned model → label
                                      + specialized Gemini 2.5 Flash → reasoning)
  → Step 4: Cross-validation
              if agree → boost confidence
              if disagree → ML tiebreaker (LogisticRegression)
  → Step 5: Final label + confidence + reasoning + feature flags
```

See [`assets/architecture.md`](assets/architecture.md) for the full Mermaid diagram.

### Key files

| File | Purpose |
|---|---|
| `main.py` | Unified entry point (4 modes) |
| `gemini_analyzer.py` | Fine-tuned / specialized Gemini analyzer |
| `agentic_pipeline.py` | 5-step orchestration pipeline |
| `fine_tune.py` | Submits fine-tuning job, saves model name |
| `finetune_data.py` | 85-example training dataset |
| `reliability_tests.py` | Automated test suite |
| `mood_analyzer.py` | Original rule-based classifier |
| `ml_experiments.py` | Original scikit-learn classifier |
| `dataset.py` | Shared labeled examples |

---

## Setup Instructions

### 1. Clone and enter the repo

```bash
git clone https://github.com/revanthBottu/ai-Integrated-system-project.git
cd ai-Integrated-system-project
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Edit `.env` and replace the placeholder:

```
GEMINI_API_KEY=your_actual_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 5. (Optional) Fine-tune the Gemini classifier

```bash
python fine_tune.py
```

This submits a supervised fine-tuning job on `gemini-1.5-flash-001-tuning` using
the 85-example dataset in `finetune_data.py`. It takes 5–30 minutes. Once done,
the model name is saved to `.tuned_model_name` and the pipeline uses it
automatically. If you skip this step, the system falls back to Gemini 2.5 Flash
with a specialized system instruction — both paths produce valid results.

### 6. Run the system

```bash
python main.py
```

Choose a mode:

```
1  Quick    — Rule-based only (offline, no API key needed)
2  Deep AI  — Full agentic pipeline with Gemini
3  Compare  — Side-by-side output from all three models
4  Test     — Reliability test suite
```

Or jump straight to a mode:

```bash
python main.py --mode 2
```

---

## Sample Interactions

### Example 1 — Sarcasm correctly caught

```
You: I absolutely love getting stuck in traffic 🙄

Text:       "I absolutely love getting stuck in traffic 🙄"
Mood:       NEGATIVE  [specialized]
Confidence: 97%
Reasoning:  The phrase uses positive language sarcastically in a clearly
            frustrating context, with a rolling-eyes emoji confirming negativity.
Detected:   sarcasm, emojis
```

### Example 2 — Mixed feelings

```
You: Proud of how far I've come but exhausted from the journey

Text:       "Proud of how far I've come but exhausted from the journey"
Mood:       MIXED  [fine-tuned]
Confidence: 88%
Reasoning:  The text simultaneously expresses pride (positive) and
            exhaustion (negative), producing a clearly mixed emotional tone.
```

### Example 3 — Slang and context

```
You: bro I'm literally cooking on this exam rn

Text:       "bro I'm literally cooking on this exam rn"
Mood:       POSITIVE  [specialized]
Confidence: 85%
Reasoning:  "Cooking" in this slang context means performing exceptionally
            well, which is a strong positive signal.
Detected:   slang
```

### Example 4 — Verbose pipeline trace

```
You: verbose
Verbose: ON

You: i'm fine. totally fine. everything is fine.

Text:       "i'm fine. totally fine. everything is fine."
Mood:       NEGATIVE  [specialized]
Confidence: 79%
Reasoning:  Repeated "totally fine" is a passive-distress pattern; the
            triple reassurance signals the opposite of the stated sentiment.

─── Pipeline trace ───
  Step 1: Input guardrail
    → Valid (45 chars)
  Step 2: Rule-based analysis
    → neutral  |  Score = 0 (positive: [], negative: [])
  Step 3: Gemini analysis (fine-tuned / specialized)
    → negative  |  confidence 79%  [specialized]
    → Reasoning: Repeated "totally fine" is a passive-distress pattern...
  Step 4: Cross-validation
    → DISAGREE: rule='neutral' gemini='negative'
    → ML tiebreaker: negative
    → ML confirms Gemini → 'negative'
  Step 5: Synthesis
    → Final: negative  |  confidence: 79%
  Latency: 1423 ms
```

---

## Design Decisions

**Why three models?**  
Each model fails differently. The rule-based model misses sarcasm and slang.
The ML model overfits to training vocabulary. Gemini misses cultural context it
wasn't primed for. Cross-validating all three catches errors that any single
model would silently make.

**Why fine-tune instead of just prompting?**  
Fine-tuning bakes our specific label schema, slang definitions, and sarcasm
rules directly into model weights rather than relying on prompt engineering that
could be overridden by the model's priors. The fine-tuned model outputs clean
labels (`positive` / `negative` / `neutral` / `mixed`) without being distracted
by adjacent reasoning tasks.

**Why keep rule-based + ML if Gemini is better?**  
Gemini API calls add latency (~1–2 s) and cost. When all three models agree,
that agreement is a reliability signal. When they disagree, the pipeline surfaces
the disagreement explicitly rather than hiding it.

**Trade-offs made:**
- Fine-tuning `gemini-1.5-flash-001-tuning` (API-accessible) vs Gemini 2.5 Flash
  (requires Vertex AI for tuning) — chosen for zero-GCP-project setup.
- Temperature = 0 for Gemini to maximize consistency at the cost of some creativity.
- ML tiebreaker only activates on disagreement, so most requests use 2 of 3 models.

---

## Testing Summary

5 test categories with the following results (baseline, before fine-tuning):

| Test | Result |
|---|---|
| Benchmark accuracy (16 examples) | ~75% |
| Consistency (3 runs per probe) | 100% consistent |
| Edge cases matched | 6/8 |
| Guardrail (oversized input) | PASS |
| Confidence calibration | High-conf predictions more accurate |

The system struggled with:
- Subtly repeated phrases ("i'm fine. totally fine.") where rule-based scored neutral
- Double negation ("not unhappy") — inconsistently labeled between models
- Single-token slang ("lol", "mid") with no surrounding context

After fine-tuning, accuracy improved on sarcasm and slang examples, which were
well-represented in the 85-example training set.

---

## Reflection

  This project taught me how to take an AI that does basic sentiment analysis, ad fine-tune it on many examples to make it really good at sentiment analysis. One other thing I did was cross-validate the AI data with the previous rule-based and ML data from the mini project, and testing the AI's reliability with unit tests. This showed me how AI responses can differ(sometimes in positive or negative ways) from heuristic methods, and how to account for and respond to those differences.

  While collaborating with AI here, I mostly created the structure of how I wanted the LLM to be fine-tuned and built into a end-to-end project, and Claude implemented it for me. I made sure to make my own tests so that the LLM wouldn't accidentally validate incorrect code. Beyond that, I was able to check the code reliability by running the website and rebuilding.

## Ethical Considerations

See [`model_card.md`](model_card.md) for the full ethics and bias analysis.
