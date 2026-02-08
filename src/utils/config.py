"""
Training configuration with dataclasses.
Includes model configs, logging, metrics export, and figure generation.
Supports both M2 Pro (Mac) and Kaggle (Linux + GPU) environments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import logging
import json
import os
import platform
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

def detect_platform() -> dict:
    """
    Detect runtime environment (M2 Pro vs Kaggle).
    
    Returns:
        dict with keys: platform, is_kaggle, has_gpu, n_cores, gpu_type
    """
    is_kaggle = os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    system = platform.system()  # Darwin (Mac), Linux, Windows
    machine = platform.machine()  # arm64 (M2), x86_64
    
    # Detect GPU
    has_cuda = False
    gpu_type = None
    
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_type = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Core count
    n_cores = os.cpu_count() or 4
    if is_kaggle:
        n_cores = min(n_cores, 4)  # Kaggle often limits to 4 cores
    
    return {
        "platform": "kaggle" if is_kaggle else system.lower(),
        "is_kaggle": is_kaggle,
        "is_mac": system == "Darwin",
        "is_arm": machine == "arm64",
        "has_gpu": has_cuda,
        "gpu_type": gpu_type,
        "n_cores": n_cores,
    }


# Platform info (cached at import time)
PLATFORM_INFO = detect_platform()


def get_device_for_catboost() -> str:
    """Get CatBoost task_type based on platform."""
    # CatBoost GPU doesn't work on Mac (no CUDA)
    if PLATFORM_INFO["has_gpu"] and not PLATFORM_INFO["is_mac"]:
        return "GPU"
    return "CPU"


def get_n_jobs() -> int:
    """Get optimal n_jobs for parallel processing."""
    return PLATFORM_INFO["n_cores"]


def get_data_dir() -> Path:
    """Get data directory based on platform."""
    if PLATFORM_INFO["is_kaggle"]:
        return Path("/kaggle/input/playground-series-s6e1")
    return Path("data/playground-series-s6e1")


def get_output_dir() -> Path:
    """Get output directory based on platform."""
    if PLATFORM_INFO["is_kaggle"]:
        return Path("/kaggle/working")
    return Path("outputs")


# =============================================================================
# LOGGER
# =============================================================================

def get_logger(
    name: str,
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get configured logger with console and optional file output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# MODEL CONFIGS (Default Values)
# =============================================================================

@dataclass
class CatBoostConfig:
    """CatBoost hyperparameters."""
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    min_data_in_leaf: int = 20
    bagging_temperature: float = 1.0
    random_seed: int = 42
    early_stopping_rounds: int = 100
    verbose: int = 100
    task_type: str = "CPU"
    
    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "min_data_in_leaf": self.min_data_in_leaf,
            "bagging_temperature": self.bagging_temperature,
            "random_seed": self.random_seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose,
            "task_type": self.task_type,
        }


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters."""
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = -1
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    # Note: early_stopping is handled via callbacks in fit(), not as a param
    early_stopping_rounds: int = 100
    
    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            # early_stopping_rounds excluded - handled via callbacks
        }


@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters."""
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 100
    verbosity: int = 0
    
    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbosity": self.verbosity,
        }


@dataclass
class MLPConfig:
    """MLP Regressor hyperparameters."""
    hidden_layer_sizes: tuple = (128, 64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: int = 256
    learning_rate_init: float = 0.001
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    random_state: int = 42
    verbose: bool = False
    
    def to_dict(self) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


# =============================================================================
# OPTUNA SEARCH SPACES
# =============================================================================
# Dataset: 630K rows, 4 numeric + 7 categorical features, target: 19.6-100

class OptunaSearchSpace:
    """
    Optuna hyperparameter search spaces for all models.
    
    Usage:
        def objective(trial):
            params = OptunaSearchSpace.catboost(trial)
            model = CatBoostRegressor(**params)
            ...
    """
    
    @staticmethod
    def catboost(trial, seed: int = 42) -> dict:
        """CatBoost search space for 630K row dataset."""
        return {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": seed,
            "early_stopping_rounds": 100,
            "verbose": 0,
            "task_type": get_device_for_catboost(),
        }
    
    @staticmethod
    def lightgbm(trial, seed: int = 42) -> dict:
        """LightGBM search space for 630K row dataset."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "n_jobs": get_n_jobs(),
            "verbose": -1,
        }
    
    @staticmethod
    def xgboost(trial, seed: int = 42) -> dict:
        """XGBoost search space for 630K row dataset."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "n_jobs": get_n_jobs(),
            "early_stopping_rounds": 100,
            "verbosity": 0,
        }
    
    @staticmethod
    def mlp(trial, seed: int = 42) -> dict:
        """MLP search space for 630K row dataset."""
        # Dynamic hidden layer architecture
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_sizes = tuple(
            trial.suggest_int(f"neurons_l{i}", 32, 256)
            for i in range(n_layers)
        )
        
        return {
            "hidden_layer_sizes": hidden_sizes,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": "adam",
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": seed,
            "verbose": False,
        }
    
    @staticmethod
    def get_search_space(model_name: str):
        """Get search space function by model name."""
        spaces = {
            "catboost": OptunaSearchSpace.catboost,
            "lightgbm": OptunaSearchSpace.lightgbm,
            "xgboost": OptunaSearchSpace.xgboost,
            "mlp": OptunaSearchSpace.mlp,
        }
        if model_name not in spaces:
            raise ValueError(f"Unknown model: {model_name}")
        return spaces[model_name]


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    """Main training configuration."""
    # Paths
    data_dir: Path = Path("data/playground-series-s6e1")
    output_dir: Path = Path("outputs")
    
    # Cross-validation
    n_folds: int = 5
    seed: int = 42
    
    # Model configs
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    
    # Logger
    _logger: logging.Logger | None = field(default=None, repr=False)
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self._setup_dirs()
    
    def _setup_dirs(self):
        """Create output directories."""
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create logger."""
        if self._logger is None:
            self._logger = get_logger(
                "train",
                log_dir=self.output_dir / "logs",
            )
        return self._logger
    
    def get_model_config(self, model_name: str) -> dict:
        """Get config dict for specific model."""
        configs = {
            "catboost": self.catboost,
            "lightgbm": self.lightgbm,
            "xgboost": self.xgboost,
            "mlp": self.mlp,
        }
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        return configs[model_name].to_dict()


# =============================================================================
# METRICS TRACKER
# =============================================================================

@dataclass
class MetricsTracker:
    """Track and export training metrics."""
    output_dir: Path = Path("outputs/metrics")
    
    results: list = field(default_factory=list)
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(
        self,
        model_name: str,
        fold: int | None = None,
        rmse: float | None = None,
        mae: float | None = None,
        r2: float | None = None,
        **extra_metrics,
    ):
        """Add a training result."""
        result = {
            "model": model_name,
            "fold": fold,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "timestamp": datetime.now().isoformat(),
            **extra_metrics,
        }
        self.results.append(result)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame with mean/std per model."""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        summary = df.groupby("model").agg({
            "rmse": ["mean", "std"],
            "mae": ["mean", "std"],
            "r2": ["mean", "std"],
        }).round(4)
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
        return summary.reset_index()
    
    def to_excel(self, filename: str = "metrics.xlsx"):
        """Export all results and summary to Excel."""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # All results
            pd.DataFrame(self.results).to_excel(
                writer, sheet_name="All Results", index=False
            )
            # Summary
            self.get_summary().to_excel(
                writer, sheet_name="Summary", index=False
            )
        
        return filepath
    
    def to_json(self, filename: str = "metrics.json"):
        """Export results to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        return filepath


# =============================================================================
# FIGURE GENERATOR
# =============================================================================

class FigureGenerator:
    """Generate training figures and plots."""
    
    def __init__(self, output_dir: Path | str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        })
    
    def plot_cv_scores(
        self,
        metrics_tracker: MetricsTracker,
        metric: str = "rmse",
        filename: str = "cv_scores.png",
    ) -> Path:
        """Plot cross-validation scores per model."""
        df = pd.DataFrame(metrics_tracker.results)
        
        fig, ax = plt.subplots()
        models = df["model"].unique()
        
        for i, model in enumerate(models):
            model_df = df[df["model"] == model]
            scores = model_df[metric].values
            folds = model_df["fold"].values
            ax.bar(
                [f + i * 0.2 for f in folds],
                scores,
                width=0.18,
                label=model,
                alpha=0.8,
            )
        
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Cross-Validation {metric.upper()} Scores")
        ax.legend()
        ax.set_xticks(range(len(df["fold"].unique())))
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
    
    def plot_model_comparison(
        self,
        metrics_tracker: MetricsTracker,
        metric: str = "rmse",
        filename: str = "model_comparison.png",
    ) -> Path:
        """Plot model comparison with mean and std."""
        summary = metrics_tracker.get_summary()
        
        fig, ax = plt.subplots()
        
        models = summary["model"]
        means = summary[f"{metric}_mean"]
        stds = summary[f"{metric}_std"]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Model Comparison - {metric.upper()}")
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
    
    def plot_learning_curve(
        self,
        train_scores: list[float],
        val_scores: list[float],
        model_name: str,
        metric: str = "rmse",
        filename: str | None = None,
    ) -> Path:
        """Plot learning curve for a model."""
        fig, ax = plt.subplots()
        
        epochs = range(1, len(train_scores) + 1)
        ax.plot(epochs, train_scores, label="Train", linewidth=2)
        ax.plot(epochs, val_scores, label="Validation", linewidth=2)
        
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Learning Curve - {model_name}")
        ax.legend()
        
        filename = filename or f"learning_curve_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
    
    def plot_feature_importance(
        self,
        feature_names: list[str],
        importances: np.ndarray,
        model_name: str,
        top_n: int = 15,
        filename: str | None = None,
    ) -> Path:
        """Plot feature importance."""
        # Sort by importance
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots()
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color=plt.cm.viridis(0.6),
            alpha=0.8,
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance - {model_name}")
        
        filename = filename or f"feature_importance_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
    
    def plot_prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        filename: str | None = None,
    ) -> Path:
        """Plot predicted vs actual values."""
        fig, ax = plt.subplots()
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")
        
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predictions vs Actual - {model_name}")
        ax.legend()
        
        filename = filename or f"pred_scatter_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        filename: str | None = None,
    ) -> Path:
        """Plot residual distribution."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")
        
        # Residual histogram
        axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")
        
        fig.suptitle(f"Residual Analysis - {model_name}")
        
        filename = filename or f"residuals_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath
