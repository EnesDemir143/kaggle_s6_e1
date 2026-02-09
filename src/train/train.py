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
# MAIN (5-FOLD CV)
# =============================================================================

def main():
    """
    Main training function with 5-Fold Cross-Validation.
    
    Features:
    - 5-Fold KFold CV
    - RAM Optimization: Train → Predict → Delete + gc.collect()
    - Fold Weighting: 1 / Fold_RMSE
    - OOF Recording for stacking/blending
    - Inter-Model Blending: 1 / Global_OOF_RMSE
    - Leakage Prevention: Fit preprocess only on train_fold
    """
    import gc
    from sklearn.model_selection import KFold
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train regression models with 5-Fold CV")
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
        help="Number of Optuna trials per model per fold",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
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
    logger.info(f"Optuna trials per fold: {args.n_trials}")
    logger.info(f"CV Folds: {args.n_folds}")
    
    # Load data
    logger.info("Loading data...")
    train_path = cfg.data_dir / "train.csv"
    test_path = cfg.data_dir / "test.csv"
    df, _ = load_data(train_path)
    test_df, _ = load_data(test_path) if test_path.exists() else (None, None)
    logger.info(f"Loaded {len(df):,} training rows")
    if test_df is not None:
        logger.info(f"Loaded {len(test_df):,} test rows")
    
    # Split features/target
    X_raw, y = split_features_target(df)
    
    # Setup KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # =========================================================================
    # STORAGE FOR OOF AND TEST PREDICTIONS
    # =========================================================================
    oof_predictions = {m: np.zeros(len(X_raw)) for m in args.models}
    test_predictions = {m: np.zeros(len(test_df)) for m in args.models} if test_df is not None else None
    fold_weights = {m: [] for m in args.models}
    
    logger.info("Initialized OOF and test prediction storage")
    
    # =========================================================================
    # 5-FOLD TRAINING LOOP
    # =========================================================================
    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        model_test_preds_weighted = np.zeros(len(test_df)) if test_df is not None else None
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
            logger.info(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
            
            # -----------------------------------------------------------------
            # 1. SPLIT DATA
            # -----------------------------------------------------------------
            X_train_fold = X_raw.iloc[train_idx].copy()
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_raw.iloc[val_idx].copy()
            y_val_fold = y.iloc[val_idx]
            
            logger.info(f"Train: {len(X_train_fold):,}, Val: {len(X_val_fold):,}")
            
            # -----------------------------------------------------------------
            # 2. PREPROCESS (fit ONLY on train_fold - NO LEAKAGE!)
            # -----------------------------------------------------------------
            X_train, fitted_objects = preprocess(
                X_train_fold,
                model=model_name,
                fit=True,
                target=y_train_fold,
            )
            X_val, _ = preprocess(
                X_val_fold,
                model=model_name,
                fit=False,
                fitted_objects=fitted_objects,
            )
            
            if test_df is not None:
                X_test, _ = preprocess(
                    test_df.copy(),
                    model=model_name,
                    fit=False,
                    fitted_objects=fitted_objects,
                )
            
            # -----------------------------------------------------------------
            # 3. TRAIN WITH OPTUNA
            # -----------------------------------------------------------------
            model, best_params, best_rmse = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train_fold,
                X_val=X_val,
                y_val=y_val_fold,
                fitted_objects=fitted_objects,
                n_trials=args.n_trials,
                logger=logger,
            )
            
            # -----------------------------------------------------------------
            # 4. PREDICT
            # -----------------------------------------------------------------
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test) if test_df is not None else None
            
            # -----------------------------------------------------------------
            # 5. CALCULATE FOLD RMSE AND WEIGHT
            # -----------------------------------------------------------------
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_weight = 1.0 / fold_rmse
            
            metrics_dict = evaluate(y_val_fold.values, val_pred)
            logger.info(f"Fold {fold + 1} - RMSE: {fold_rmse:.4f}, Weight: {fold_weight:.4f}")
            
            # Track metrics per fold
            metrics.add_result(
                model_name=model_name,
                fold=fold + 1,
                **metrics_dict,
                best_params=best_params,
            )
            
            # -----------------------------------------------------------------
            # 6. STORE OOF AND WEIGHTED TEST PREDICTIONS
            # -----------------------------------------------------------------
            oof_predictions[model_name][val_idx] = val_pred
            if test_df is not None and test_pred is not None:
                model_test_preds_weighted += test_pred * fold_weight
            fold_weights[model_name].append(fold_weight)
            
            # -----------------------------------------------------------------
            # 7. RAM CLEANUP (CRITICAL!)
            # -----------------------------------------------------------------
            del model, X_train, X_val, fitted_objects
            del X_train_fold, X_val_fold, val_pred
            if test_df is not None:
                del X_test, test_pred
            gc.collect()
            logger.info(f"RAM cleaned after fold {fold + 1}")
        
        # ---------------------------------------------------------------------
        # NORMALIZE TEST PREDICTIONS BY TOTAL FOLD WEIGHT
        # ---------------------------------------------------------------------
        total_weight = sum(fold_weights[model_name])
        if test_df is not None:
            test_predictions[model_name] = model_test_preds_weighted / total_weight
        
        logger.info(f"\n{model_name.upper()} Complete! Total fold weight: {total_weight:.4f}")
    
    # =========================================================================
    # CALCULATE GLOBAL OOF METRICS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("OOF RESULTS")
    logger.info("="*60)
    
    oof_results = {}
    for model_name in args.models:
        oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions[model_name]))
        oof_mae = mean_absolute_error(y, oof_predictions[model_name])
        oof_r2 = r2_score(y, oof_predictions[model_name])
        
        oof_results[model_name] = {
            "oof_rmse": oof_rmse,
            "oof_mae": oof_mae,
            "oof_r2": oof_r2,
        }
        
        logger.info(f"{model_name.upper()}: OOF RMSE={oof_rmse:.4f}, MAE={oof_mae:.4f}, R²={oof_r2:.4f}")
    
    # =========================================================================
    # INTER-MODEL BLENDING (1/OOF_RMSE WEIGHTS)
    # =========================================================================
    model_weights = {m: 1.0 / oof_results[m]["oof_rmse"] for m in args.models}
    total_model_weight = sum(model_weights.values())
    normalized_weights = {m: w / total_model_weight for m, w in model_weights.items()}
    
    logger.info("\n" + "="*60)
    logger.info("INTER-MODEL WEIGHTS (1/OOF_RMSE)")
    logger.info("="*60)
    for m in args.models:
        logger.info(f"  {m}: {normalized_weights[m]:.4f} ({normalized_weights[m]*100:.1f}%)")
    
    # Blend OOF predictions
    final_oof_pred = sum(
        oof_predictions[m] * normalized_weights[m]
        for m in args.models
    )
    blend_metrics = evaluate(y.values, final_oof_pred)
    
    logger.info(f"\nBLENDED OOF: RMSE={blend_metrics['rmse']:.4f}, "
               f"MAE={blend_metrics['mae']:.4f}, R²={blend_metrics['r2']:.4f}")
    
    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================
    output_dir = cfg.output_dir
    
    # Save OOF predictions
    oof_dir = output_dir / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in args.models:
        np.save(oof_dir / f"{model_name}_oof.npy", oof_predictions[model_name])
        logger.info(f"Saved OOF predictions: {oof_dir / f'{model_name}_oof.npy'}")
    
    # Save blended OOF
    np.save(oof_dir / "blended_oof.npy", final_oof_pred)
    
    # Save test predictions if available
    if test_df is not None and test_predictions is not None:
        # Blend test predictions
        final_test_pred = sum(
            test_predictions[m] * normalized_weights[m]
            for m in args.models
        )
        
        test_dir = output_dir / "test_predictions"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name in args.models:
            np.save(test_dir / f"{model_name}_test.npy", test_predictions[model_name])
        np.save(test_dir / "blended_test.npy", final_test_pred)
        
        logger.info(f"Saved test predictions to {test_dir}")
    
    # Save model weights
    weights_path = output_dir / "model_weights.json"
    import json
    with open(weights_path, "w") as f:
        json.dump({
            "normalized_weights": normalized_weights,
            "oof_results": oof_results,
            "blend_metrics": blend_metrics,
        }, f, indent=2)
    logger.info(f"Saved model weights: {weights_path}")
    
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

