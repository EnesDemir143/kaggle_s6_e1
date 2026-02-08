# Preprocessing Guide for Regression Models

This document explains the preprocessing requirements for each model type used in this project.

---

## Model Overview

| Model | Normalization | Categorical Encoding | Notes |
|-------|---------------|---------------------|-------|
| **CatBoost** | ❌ Not needed | ❌ Native support | Handles everything internally |
| **LightGBM** | ❌ Not needed | ⚠️ Category dtype | Convert to `category` type |
| **XGBoost** | ❌ Not needed | ✅ Required | Use ordinal/label/target |
| **MLP** | ✅ Required | ✅ Required | Use standard + onehot |

---

## CatBoost

**Preprocessing:** None required

CatBoost natively handles categorical features without any encoding. It uses an advanced algorithm called "ordered target encoding" internally.

```python
X, fitted = preprocess(df, model="catboost", fit=True)
```

**Why no preprocessing?**
- Built-in categorical feature handling
- Automatic missing value handling
- No normalization needed (tree-based)

---

## LightGBM

**Preprocessing:** Convert categorical columns to `category` dtype

LightGBM can handle categorical features natively if they are converted to pandas `category` type.

```python
X, fitted = preprocess(df, model="lightgbm", fit=True)
```

**Why just dtype conversion?**
- Native categorical support with `category` dtype
- Uses histogram-based algorithm
- Faster than one-hot encoding
- No normalization needed (tree-based)

---

## XGBoost

**Preprocessing:** Categorical encoding required

XGBoost requires all features to be numeric. Categorical features must be encoded.

```python
# Ordinal encoding (default, fastest)
X, fitted = preprocess(df, model="xgboost", fit=True, enc_method="ordinal")

# Target encoding (often better performance)
X, fitted = preprocess(df, model="xgboost", fit=True, enc_method="target", target=y)
```

**Encoding options:**
| Method | Best for |
|--------|----------|
| `ordinal` | Default, fast, low memory |
| `label` | Simple integer encoding |
| `target` | High cardinality features |

**Why no normalization?**
- Tree-based models split on thresholds
- Scale doesn't affect split decisions

---

## MLP Regressor

**Preprocessing:** Both normalization and encoding required

Neural networks are sensitive to feature scales and require numeric inputs.

```python
# Standard normalization + OneHot encoding (recommended)
X, fitted = preprocess(df, model="mlp", fit=True, norm_method="standard")

# MinMax normalization (for [0,1] range)
X, fitted = preprocess(df, model="mlp", fit=True, norm_method="minmax")
```

**Why normalization is critical?**
- Gradient descent converges faster with normalized features
- Prevents features with large values from dominating
- Improves training stability

**Why OneHot encoding?**
- MLP treats inputs as continuous values
- Ordinal encoding implies false ordering (e.g., male=0 < female=1)
- OneHot preserves categorical nature

**Normalization options:**
| Method | Formula | Best for |
|--------|---------|----------|
| `standard` | `(x - μ) / σ` | General use, assumes normal distribution |
| `minmax` | `(x - min) / (max - min)` | Bounded data, sigmoid/tanh activations |
| `robust` | Uses median/IQR | Data with outliers |

---

## Quick Reference

```python
from src.utils.preprocess import preprocess, split_features_target, load_data

# Load data
train_df, test_df = load_data("data/train.csv", "data/test.csv")
X_train, y_train = split_features_target(train_df)

# Preprocess for training
X_train_processed, fitted = preprocess(X_train, model="xgboost", fit=True)

# Preprocess for inference (use fitted objects)
X_test_processed, _ = preprocess(test_df, model="xgboost", fit=False, fitted_objects=fitted)
```

---

## Important Notes

1. **Always use `fit=True` for training data** - This learns the parameters (mean, std, categories)
2. **Always use `fit=False` for test data** - This prevents data leakage
3. **Save fitted objects for production** - Use `save_fitted_objects()` function
4. **Target encoding needs target variable** - Pass `target=y` parameter
