# Mood Machine — System Architecture

```mermaid
flowchart TD
    A([User Input]) --> B{Step 1\nInput Guardrail}

    B -->|"Empty / too long"| C([⛔ Blocked\nneutral + error])
    B -->|Valid| D[Step 2\nRule-Based Analyzer\nmood_analyzer.py]

    D --> E[Step 3\nGemini Analyzer\ngemini_analyzer.py]

    subgraph E_box[" Gemini Analyzer (gemini_analyzer.py)"]
        direction TB
        FT{Fine-tuned\nmodel available?}
        FT -->|Yes| FTM[Fine-tuned\ngemini-1.5-flash\noutputs label only]
        FT -->|No| BASE[Specialized\nGemini 2.5 Flash\noutputs label + JSON]
        FTM --> MERGE[Merge:\nlabel from fine-tuned\nconfidence+reasoning\nfrom base model]
        BASE --> MERGE
    end

    D --> CV
    E_box --> CV

    CV{Step 4\nCross-Validation}

    CV -->|"Rule == Gemini\n(agree)"| BOOST[Boost confidence +10%]
    CV -->|"Gemini = mixed\n≥ 65% conf"| TRUST[Trust Gemini\n(only model\nfor 'mixed')"]
    CV -->|Disagree| ML[ML Tiebreaker\nLogisticRegression\nml_experiments.py]

    ML -->|ML = Gemini| CONF[Use Gemini label]
    ML -->|ML = Rule| RULE[Use Rule label]
    ML -->|All differ| DEFER[Defer to Gemini\n−10% confidence]

    BOOST --> SYN
    TRUST --> SYN
    CONF --> SYN
    RULE --> SYN
    DEFER --> SYN

    SYN([Step 5\nFinal Result\nlabel · confidence\nreasoning · feature flags\nstep trace])

    subgraph TESTS["Reliability Layer (reliability_tests.py)"]
        T1[Benchmark\nvs TRUE_LABELS]
        T2[Consistency\n3 runs same input]
        T3[Edge Cases\nemoji-only, caps, etc]
        T4[Guardrail Test\noversized input]
        T5[Calibration\nhigh-conf = higher acc]
    end

    SYN -.->|evaluated by| TESTS
```

## Component Descriptions

| Component | File | Role |
|---|---|---|
| Input Guardrail | `agentic_pipeline.py` | Blocks empty or oversized inputs before any API call |
| Rule-Based Analyzer | `mood_analyzer.py` | Fast, transparent word-score classifier with negation & sarcasm heuristics |
| Fine-tuned Model | `fine_tune.py` / `gemini_analyzer.py` | `gemini-1.5-flash-001-tuning` trained on 85 mood-labeled examples |
| Specialized Base Model | `gemini_analyzer.py` | `gemini-2.5-flash-preview-05-20` with system instruction encoding label schema + slang/sarcasm rules |
| ML Tiebreaker | `ml_experiments.py` | LogisticRegression on bag-of-words features, used only on disagreement |
| Agentic Pipeline | `agentic_pipeline.py` | Orchestrates all steps, resolves disagreements, builds final result |
| Reliability Tests | `reliability_tests.py` | Benchmark, consistency, edge cases, guardrail, and calibration tests |

## Data Flow

```
User text
  → guardrail check
  → rule-based scoring (local)
  → Gemini analysis (fine-tuned label + base reasoning)
  → cross-validation against rule-based
      ↳ on disagreement: ML tiebreaker
  → final label + confidence + reasoning + feature flags
```
