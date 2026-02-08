# Evaluation Guide

This document explains how to generate predictions and Kaggle submissions.

---

## Quick Start

```bash
# Single model prediction
uv run python -m src.evaluate.evaluate --model xgboost

# All trained models
uv run python -m src.evaluate.evaluate --model all

# Ensemble prediction
uv run python -m src.evaluate.evaluate --model ensemble --ensemble-method mean
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `all` | `catboost`, `lightgbm`, `xgboost`, `mlp`, `all`, or `ensemble` |
| `--ensemble-method` | `mean` | `mean` or `weighted` |
| `--weights` | equal | Weights for weighted ensemble (space-separated) |
| `--output` | auto | Custom output filename |

---

## Examples

```bash
# Weighted ensemble (CatBoost 40%, LightGBM 30%, XGBoost 20%, MLP 10%)
uv run python -m src.evaluate.evaluate \
  --model ensemble \
  --ensemble-method weighted \
  --weights 0.4 0.3 0.2 0.1

# Custom output name
uv run python -m src.evaluate.evaluate --model xgboost --output my_best_submission
```

---

## Output Files

Submissions are saved to `outputs/submissions/`:

```
outputs/submissions/
├── xgboost_20260208_181358.csv
├── ensemble_mean_20260208_182000.csv
└── my_best_submission_20260208_183000.csv
```

**Format:** Kaggle-compatible CSV with `id` and `exam_score` columns.

---

## Data Flow

```
test.csv
    ↓
load_test_data()
    ↓
preprocess(fit=False, fitted_objects=...)  ← No data leakage!
    ↓
model.predict()
    ↓
submission.csv
```

> [!IMPORTANT]
> Always use `fit=False` with pre-fitted objects to prevent data leakage.

---

## Related Docs

- [Training Guide](training_guide.md) - How to train models
- [Preprocessing Guide](preprocessing_guide.md) - Preprocessing details
