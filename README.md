# 🎓 Student Test Score Prediction

Kaggle Playground Series S6E1 competition solution using ensemble of gradient boosting and neural network models.

---

## 📋 Overview

| Aspect | Details |
|--------|---------|
| **Competition** | [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1) |
| **Task** | Regression - Predict student exam scores |
| **Dataset** | 630K train / 270K test samples |
| **Features** | 4 numeric + 7 categorical |
| **Target** | `exam_score` (19.6 - 100) |

---

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Train models
uv run python -m src.train.train

# Generate submissions
uv run python -m src.evaluate.evaluate --model all
```

---

## 📁 Project Structure

```
├── data/
│   └── playground-series-s6e1/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── src/
│   ├── train/          # Training module
│   ├── evaluate/       # Evaluation & submission
│   └── utils/          # Config, preprocessing
├── outputs/
│   ├── models/         # Trained models
│   ├── submissions/    # Kaggle submissions
│   ├── figures/        # Visualizations
│   └── metrics/        # Performance metrics
├── docs/               # Documentation
└── notebooks/          # Exploration notebooks
```

---

## 🤖 Models

| Model | Type | Description |
|-------|------|-------------|
| **CatBoost** | Gradient Boosting | Native categorical handling |
| **LightGBM** | Gradient Boosting | Fast histogram-based |
| **XGBoost** | Gradient Boosting | Robust, widely used |
| **MLP** | Neural Network | Multi-layer perceptron |

All models use **Optuna** for hyperparameter optimization.

---

## 📚 Documentation

| Doc | Description |
|-----|-------------|
| [Training Guide](docs/training_guide.md) | How to train models |
| [Evaluation Guide](docs/evaluation_guide.md) | Generate predictions & submissions |
| [Preprocessing Guide](docs/preprocessing_guide.md) | Data preprocessing details |

---

## 🔧 CLI Reference

### Training

```bash
uv run python -m src.train.train [OPTIONS]

Options:
  --models      Models to train (default: all)
  --n-trials    Optuna trials (default: 50)
  --val-size    Validation size (default: 0.1)
```

### Evaluation

```bash
uv run python -m src.evaluate.evaluate [OPTIONS]

Options:
  --model            Model or "all" or "ensemble"
  --ensemble-method  "mean" or "weighted"
  --weights          Weights for weighted ensemble
  --output           Custom output filename
```

---

## 📊 Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | numeric | Student age |
| `study_hours` | numeric | Weekly study hours |
| `class_attendance` | numeric | Attendance percentage |
| `sleep_hours` | numeric | Daily sleep hours |
| `gender` | categorical | Student gender |
| `course` | categorical | Course type |
| `internet_access` | categorical | Has internet |
| `sleep_quality` | categorical | Sleep quality rating |
| `study_method` | categorical | Study method used |
| `facility_rating` | categorical | School facility rating |
| `exam_difficulty` | categorical | Exam difficulty level |

---

## 📄 License

MIT
