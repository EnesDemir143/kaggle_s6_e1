"""
Training script for student test score prediction.
Trains 4 models (CatBoost, LightGBM, XGBoost, MLP) with Optuna hyperparameter tuning.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Model imports
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Local imports
from src.utils.config import (
    TrainConfig,
    MetricsTracker,
    FigureGenerator,
    OptunaSearchSpace,
    get_data_dir,
    get_output_dir,
    PLATFORM_INFO,
)
from src.utils.preprocess import (
    load_data,
    preprocess,
    split_features_target,
    save_fitted_objects,
    CATEGORICAL_COLUMNS,
)


# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = ["catboost", "lightgbm", "xgboost", "mlp"]
RANDOM_STATE = 42


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_name: str, params: dict):
    """Create model instance with given parameters."""
    match model_name:
        case "catboost":
            return CatBoostRegressor(**params)
        case "lightgbm":
            return LGBMRegressor(**params)
        case "xgboost":
            return XGBRegressor(**params)
        case "mlp":
            return MLPRegressor(**params)
        case _:
            raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fitted_objects: dict,
) -> float:
    """Optuna objective function for hyperparameter tuning."""
    
    # Get search space
    search_space_fn = OptunaSearchSpace.get_search_space(model_name)
    params = search_space_fn(trial, seed=RANDOM_STATE)
    
    # Create and train model
    model = create_model(model_name, params)
    
    match model_name:
        case "catboost":
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=fitted_objects.get("cat_features", CATEGORICAL_COLUMNS),
                use_best_model=True,
                verbose=0,
            )
        case "lightgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                    optuna.integration.LightGBMPruningCallback(trial, "rmse"),
                ],
            )
        case "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        case "mlp":
            model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fitted_objects: dict,
    n_trials: int = 50,
    logger=None,
) -> tuple:
    """Train model with Optuna hyperparameter tuning."""
    
    if logger:
        logger.info(f"Starting Optuna study for {model_name} ({n_trials} trials)")
    
    # Create Optuna study with fixed sampler seed for reproducibility
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_name}_study",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, model_name, X_train, y_train, X_val, y_val, fitted_objects
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential for stability
    )
    
    if logger:
        logger.info(f"Best trial: {study.best_trial.number}, RMSE: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
    
    # Retrain with best params
    search_space_fn = OptunaSearchSpace.get_search_space(model_name)
    
    # Reconstruct best params from trial
    best_params = search_space_fn(study.best_trial, seed=RANDOM_STATE)
    
    model = create_model(model_name, best_params)
    
    match model_name:
        case "catboost":
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=fitted_objects.get("cat_features", CATEGORICAL_COLUMNS),
                use_best_model=True,
                verbose=100,
            )
        case "lightgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
        case "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        case "mlp":
            model.fit(X_train, y_train)
    
    return model, study.best_params, study.best_value


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_artifacts(
    model,
    model_name: str,
    fitted_objects: dict,
    output_dir: Path,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    fig_gen: FigureGenerator,
    logger=None,
) -> None:
    """Save model, preprocessors, and figures."""
    
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / f"{model_name}_best.joblib"
    joblib.dump(model, model_path)
    if logger:
        logger.info(f"Saved model: {model_path}")
    
    # Save fitted objects
    fitted_path = models_dir / f"{model_name}_fitted_objects.joblib"
    save_fitted_objects(fitted_objects, fitted_path)
    if logger:
        logger.info(f"Saved preprocessors: {fitted_path}")
    
    # Generate figures
    # 1. Prediction scatter
    fig_gen.plot_prediction_scatter(y_val, y_pred, model_name)
    
    # 2. Residuals
    fig_gen.plot_residuals(y_val, y_pred, model_name)
    
    # 3. Feature importance (tree models only)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig_gen.plot_feature_importance(feature_names, importances, model_name)
    
    if logger:
        logger.info(f"Saved figures for {model_name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train regression models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Models to train",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size (fraction of training data)",
    )
    args = parser.parse_args()
    
    # Setup
    cfg = TrainConfig(
        data_dir=get_data_dir(),
        output_dir=get_output_dir(),
    )
    logger = cfg.logger
    metrics = MetricsTracker(output_dir=cfg.output_dir / "metrics")
    fig_gen = FigureGenerator(output_dir=cfg.output_dir / "figures")
    
    logger.info(f"Platform: {PLATFORM_INFO['platform']}")
    logger.info(f"GPU available: {PLATFORM_INFO['has_gpu']}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"Optuna trials: {args.n_trials}")
    
    # Load data
    logger.info("Loading data...")
    train_path = cfg.data_dir / "train.csv"
    df, _ = load_data(train_path)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Split features/target
    X, y = split_features_target(df)
    
    # Train/val split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_size,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Train: {len(X_train_raw):,}, Val: {len(X_val_raw):,}")
    
    # Train each model
    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Preprocess - fit on train only
        X_train, fitted_objects = preprocess(
            X_train_raw.copy(),
            model=model_name,
            fit=True,
            target=y_train,
        )
        
        # Transform validation with fitted objects
        X_val, _ = preprocess(
            X_val_raw.copy(),
            model=model_name,
            fit=False,
            fitted_objects=fitted_objects,
        )
        
        # Train with Optuna
        model, best_params, best_rmse = train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fitted_objects=fitted_objects,
            n_trials=args.n_trials,
            logger=logger,
        )
        
        # Final evaluation
        y_pred = model.predict(X_val)
        metrics_dict = evaluate(y_val.values, y_pred)
        
        logger.info(f"Final metrics - RMSE: {metrics_dict['rmse']:.4f}, "
                   f"MAE: {metrics_dict['mae']:.4f}, R²: {metrics_dict['r2']:.4f}")
        
        # Track metrics
        metrics.add_result(
            model_name=model_name,
            fold=None,  # No CV
            **metrics_dict,
            best_params=best_params,
        )
        
        # Save artifacts
        save_artifacts(
            model=model,
            model_name=model_name,
            fitted_objects=fitted_objects,
            output_dir=cfg.output_dir,
            y_val=y_val.values,
            y_pred=y_pred,
            feature_names=list(X_train.columns),
            fig_gen=fig_gen,
            logger=logger,
        )
    
    # Export metrics
    excel_path = metrics.to_excel("training_metrics.xlsx")
    json_path = metrics.to_json("training_metrics.json")
    logger.info(f"\nMetrics exported to:")
    logger.info(f"  - {excel_path}")
    logger.info(f"  - {json_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    summary = metrics.get_summary()
    if not summary.empty:
        logger.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
