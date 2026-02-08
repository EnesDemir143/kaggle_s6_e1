# Training Guide

This document explains how to train regression models for student test score prediction.

---

## Quick Start

```bash
# Train all models with default settings
uv run python -m src.train.train

# Train specific models
uv run python -m src.train.train --models catboost xgboost

# Adjust hyperparameter search trials
uv run python -m src.train.train --n-trials 100
```

---

## Available Models

| Model | Type | GPU Support | Best For |
|-------|------|-------------|----------|
| **CatBoost** | Gradient Boosting | ✅ CUDA | Categorical-heavy data |
| **LightGBM** | Gradient Boosting | ✅ CUDA | Large datasets, speed |
| **XGBoost** | Gradient Boosting | ✅ CUDA | General performance |
| **MLP** | Neural Network | ❌ CPU only | Non-linear patterns |

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | all | Models to train: `catboost`, `lightgbm`, `xgboost`, `mlp` |
| `--n-trials` | 50 | Optuna hyperparameter search trials |
| `--val-size` | 0.1 | Validation set fraction |

---

## Output Files

Training produces these outputs in `outputs/`:

```
outputs/
├── models/
│   ├── {model}_best.joblib           # Trained model
│   └── {model}_fitted_objects.joblib # Preprocessing objects
├── figures/
│   ├── pred_scatter_{model}.png      # Predictions vs actual
│   ├── residuals_{model}.png         # Residual analysis
│   └── feature_importance_{model}.png # Feature importance
├── metrics/
│   ├── training_metrics.json
│   └── training_metrics.xlsx
└── logs/
    └── train_{timestamp}.log
```

---

## Hyperparameter Tuning

Optuna is used for hyperparameter optimization with TPE sampler.

### Search Spaces

Defined in `src/utils/config.py` → `OptunaSearchSpace` class.

| Model | Key Parameters |
|-------|----------------|
| CatBoost | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg` |
| LightGBM | `n_estimators`, `num_leaves`, `subsample`, `reg_alpha/lambda` |
| XGBoost | `n_estimators`, `max_depth`, `min_child_weight`, `gamma` |
| MLP | `hidden_layer_sizes`, `alpha`, `batch_size`, `learning_rate_init` |

---

## Related Docs

- [Preprocessing Guide](preprocessing_guide.md) - Data preprocessing details
- [Evaluation Guide](evaluation_guide.md) - Test predictions & submissions
