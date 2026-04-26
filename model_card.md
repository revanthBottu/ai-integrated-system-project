# Model Card: Mood Machine — AI Integrated System

This model card covers the expanded Mood Machine, which now includes three
classifiers: the original rule-based model, the original ML model, and a
fine-tuned / specialized Gemini classifier orchestrated by an agentic pipeline.

---

## 1. Model Overview

**Models used:**  
1. Rule-based classifier (`mood_analyzer.py`) — hand-crafted scoring rules  
2. ML classifier (`ml_experiments.py`) — scikit-learn LogisticRegression  
3. Gemini classifier (`gemini_analyzer.py`) — fine-tuned `gemini-1.5-flash-001-tuning`
   (or `gemini-2.5-flash-preview-05-20` with specialized system instruction as fallback)

**Intended purpose:**  
Classify short social-media posts and messages into one of four mood labels:
`positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**  
The agentic pipeline runs the rule-based and Gemini classifiers in parallel,
cross-validates their predictions, and uses the ML model as a tiebreaker when
they disagree. All three models were designed to complement each other's failure
modes: the rule-based model is transparent but literal; the ML model learns
from labeled examples; Gemini understands sarcasm, slang, and context.

---

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` (16 hand-labeled examples) is used to evaluate the rule-based and
ML models. `finetune_data.py` contains 85 examples used to fine-tune the Gemini
classifier, covering positive, negative, neutral, and mixed labels with broad
coverage of sarcasm, negation, slang, emojis, and ambiguous phrasing.

**Labeling process:**  
Labels were assigned by the developer based on personal interpretation of tone.
Several examples were intentionally hard to label (e.g., "i'm fine. totally fine.
everything is fine." — labeled negative due to the passive-distress pattern of
repeated reassurance). Some labels could legitimately differ between annotators.

**Important characteristics of the dataset:**
- Contains Internet slang (cooking, buns, W, L, mid, no cap, lowkey)
- Includes sarcasm (positive words in negative contexts)
- Includes emojis (both positive and negative)
- Includes negation and double negation
- Includes mixed-feeling sentences
- English only; single-developer annotations

**Possible issues with the dataset:**
- Small size (85 fine-tuning examples, 16 evaluation examples)
- Single-annotator bias — another person might label some examples differently
- Imbalance: negative examples are over-represented in the fine-tuning set
- No held-out test set — evaluation accuracy is on training-adjacent data

---

## 3. Rule-Based Model

**Scoring rules:**
- Positive words → +1; Negative words → −1
- Emoji tokens → ±2
- Negation words flip the sign of the next scored token
- Positive words near "negative situation" words (traffic, stuck, fail) → treated as sarcasm → −1

**Strengths:** Transparent, fast, no API needed, reproducible.  
**Weaknesses:** Fails on sarcasm outside the hard-coded situation-word list,
unfamiliar slang, and any phrasing it has no rules for.

---

## 4. ML Model

**Features:** Bag-of-words using `CountVectorizer`.  
**Training data:** `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.  
**Strengths:** Learns vocabulary patterns automatically; handles some slang.  
**Weaknesses:** Overfits to the tiny training set; can't generalize to novel slang
or out-of-vocabulary words; never produces `mixed` reliably with only a few examples.

---

## 5. Gemini Fine-Tuned / Specialized Model

**Fine-tuning approach:**  
A supervised fine-tuning job is submitted to the Gemini API using
`gemini-1.5-flash-001-tuning` as the source model and the 85-example dataset
from `finetune_data.py`. Training data maps text → bare label. The model learns
the exact label schema and our dataset's linguistic patterns.

**Fallback (specialized system instruction):**  
When the fine-tuned model is not available, `gemini-2.5-flash-preview-05-20` is
used with a detailed system instruction encoding the label definitions, sarcasm
rule, negation rule, and a slang vocabulary. This is a legitimate form of model
specialization: the model's default behavior is constrained to our task.

**Output:** The Gemini component always produces a label, confidence score,
one-sentence reasoning, and feature flags (sarcasm, negation, slang, emojis).

---

## 6. Evaluation

**Benchmark accuracy:** ~75% on the 16 labeled examples in `SAMPLE_POSTS`
(before fine-tuning; improves with fine-tuned model on sarcasm and slang cases).

**Consistency:** Across 3 runs on the same input (with temperature = 0),
the fine-tuned and specialized models produce identical output.

**Examples of correct predictions:**
- `"I absolutely love getting stuck in traffic 🙄"` → negative
  (Gemini correctly identifies sarcasm + eye-roll emoji)
- `"I can't wait to waste my time on some gardening."` → negative
  (negation + "waste" signal caught by Gemini and rule-based)
- `"I'm cooking on this project rn"` → positive
  (fine-tuned model learned "cooking" = performing well)

**Examples of incorrect / challenging predictions:**
- `"I am not happy about this"` → rule-based scores positive (misses negation);
  Gemini correctly says negative; ML tiebreaker resolves to negative ✓
- `"lol"` → hard to classify; model returns neutral, but could be positive or negative
- `"i'm fine. totally fine. everything is fine."` → rule-based neutral, Gemini negative;
  ML confirms negative, final answer negative ✓

---

## 7. Limitations

- **Small dataset** — 85 examples is very limited for fine-tuning; the model
  likely does not generalize to slang coined after the training data was written.
- **English only** — no support for other languages or code-switching.
- **Single-sentence inputs** — long multi-sentence posts may confuse the pipeline.
- **Sarcasm heuristics** — the rule-based sarcasm detection relies on a fixed list
  of "negative situation words"; sarcasm in other contexts is missed.
- **No true test set** — accuracy is measured on training-adjacent examples.
- **API dependency** — Modes 2, 3, and 4 require a valid Gemini API key and
  internet access; network errors cause graceful fallback but reduce accuracy.

---

## 8. Ethical Considerations

**Misclassifying distress:** If deployed in a mental-health or support context,
a neutral or positive classification of a message expressing distress (e.g., "i'm
fine. totally fine.") could cause a system to ignore someone who needs help.
Guardrails, human review, and a low confidence threshold for "neutral" outputs
should be required before any such deployment.

**Misuse potential:**
- The system could be used to auto-triage user-generated content at scale,
  potentially silencing negative feedback by labeling it as noise.
- Prevention: require human review for any moderation decision, never use a single
  model's output as a final decision, and publish the model card alongside the system.

**Bias toward developer's dialect:** All training data was written or selected by
one English speaker. The model may perform poorly for speakers of African American
Vernacular English, non-US English slang, or non-standard orthography.

---

## 9. Model Reflections

Passed most test cases, but failed on some more neutral test cases, and struggles sometimes labeling as neutral, instead jumps the gun to positive or negative.

  The limitations are it relies on knowing the current slang of the speaker. If it's a different dialect or new slang, the LLM might have bias towards older or more prevalent slang. This would lead to more obscure dictionaries of Engish slang working not as good.

  What surprised me was how often the rule-based model disagreed. I can probably attribute this to the LLM having a more diverse training dataset, but in a lot of cases the ML method had to tie break, and more weight had to be assigned to the LLM response.
