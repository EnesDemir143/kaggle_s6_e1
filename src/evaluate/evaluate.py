"""
Evaluation script for test data predictions.
Generates Kaggle submissions from trained models.
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import joblib

# Local imports
from src.utils.config import (
    get_data_dir,
    get_output_dir,
    MetricsTracker,
    get_logger,
)
from src.utils.preprocess import (
    load_data,
    preprocess,
    load_fitted_objects,
    ID_COLUMN,
    TARGET_COLUMN,
)


# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = ["catboost", "lightgbm", "xgboost", "mlp"]
EnsembleMethod = Literal["mean", "weighted"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test.csv and sample_submission.csv.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        (test_df, sample_submission_df)
    """
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "sample_submission.csv"
    
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(submission_path)
    
    return test_df, sample_df


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_available_models(models_dir: Path) -> list[str]:
    """Get list of trained models in models directory."""
    available = []
    for model_name in MODELS:
        model_path = models_dir / f"{model_name}_best.joblib"
        fitted_path = models_dir / f"{model_name}_fitted_objects.joblib"
        if model_path.exists() and fitted_path.exists():
            available.append(model_name)
    return available


def load_model_artifacts(
    model_name: str,
    models_dir: Path,
) -> tuple[object, dict]:
    """
    Load trained model and fitted preprocessing objects.
    
    Args:
        model_name: Name of the model
        models_dir: Path to models directory
    
    Returns:
        (model, fitted_objects)
    """
    model_path = models_dir / f"{model_name}_best.joblib"
    fitted_path = models_dir / f"{model_name}_fitted_objects.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not fitted_path.exists():
        raise FileNotFoundError(f"Fitted objects not found: {fitted_path}")
    
    model = joblib.load(model_path)
    fitted_objects = load_fitted_objects(fitted_path)
    
    return model, fitted_objects


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_model(
    model_name: str,
    test_df: pd.DataFrame,
    models_dir: Path,
    logger=None,
) -> np.ndarray:
    """
    Make predictions for a single model.
    
    Args:
        model_name: Name of the model
        test_df: Test DataFrame
        models_dir: Path to models directory
        logger: Optional logger
    
    Returns:
        Predictions array
    """
    if logger:
        logger.info(f"Loading model: {model_name}")
    
    # Load model and fitted objects
    model, fitted_objects = load_model_artifacts(model_name, models_dir)
    
    # Preprocess test data (fit=False to use pre-fitted objects)
    X_test, _ = preprocess(
        test_df.copy(),
        model=model_name,
        fit=False,
        fitted_objects=fitted_objects,
    )
    
    if logger:
        logger.info(f"Preprocessed test data: {X_test.shape}")
    
    # Predict
    predictions = model.predict(X_test)
    
    if logger:
        logger.info(f"Predictions - min: {predictions.min():.2f}, max: {predictions.max():.2f}")
    
    return predictions


def ensemble_predict(
    model_names: list[str],
    test_df: pd.DataFrame,
    models_dir: Path,
    method: EnsembleMethod = "mean",
    weights: list[float] | None = None,
    logger=None,
) -> np.ndarray:
    """
    Make ensemble predictions from multiple models.
    
    Args:
        model_names: List of model names
        test_df: Test DataFrame
        models_dir: Path to models directory
        method: Ensemble method ("mean" or "weighted")
        weights: Optional weights for weighted average
        logger: Optional logger
    
    Returns:
        Ensemble predictions array
    """
    if logger:
        logger.info(f"Ensemble prediction with {len(model_names)} models, method: {method}")
    
    all_predictions = []
    
    for model_name in model_names:
        preds = predict_single_model(model_name, test_df, models_dir, logger)
        all_predictions.append(preds)
    
    predictions_matrix = np.column_stack(all_predictions)
    
    if method == "mean":
        ensemble_preds = predictions_matrix.mean(axis=1)
    elif method == "weighted":
        if weights is None:
            # Equal weights if not specified
            weights = [1.0 / len(model_names)] * len(model_names)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        ensemble_preds = (predictions_matrix * weights).sum(axis=1)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    if logger:
        logger.info(f"Ensemble predictions - min: {ensemble_preds.min():.2f}, max: {ensemble_preds.max():.2f}")
    
    return ensemble_preds


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================

def generate_submission(
    ids: pd.Series,
    predictions: np.ndarray,
    output_dir: Path,
    model_name: str,
    logger=None,
) -> Path:
    """
    Generate Kaggle submission file.
    
    Args:
        ids: ID column from test data
        predictions: Model predictions
        output_dir: Output directory
        model_name: Name for the submission file
        logger: Optional logger
    
    Returns:
        Path to submission file
    """
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        ID_COLUMN: ids,
        TARGET_COLUMN: predictions,
    })
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.csv"
    filepath = submissions_dir / filename
    
    # Save
    submission_df.to_csv(filepath, index=False)
    
    if logger:
        logger.info(f"Submission saved: {filepath}")
        logger.info(f"  Rows: {len(submission_df):,}")
        logger.info(f"  Columns: {list(submission_df.columns)}")
    
    return filepath


# =============================================================================
# METRICS (OPTIONAL)
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    metrics_tracker: MetricsTracker,
    logger=None,
) -> dict:
    """
    Calculate and log metrics if ground truth is available.
    
    Args:
        y_true: Ground truth values
        y_pred: Predictions
        model_name: Model name for logging
        metrics_tracker: MetricsTracker instance
        logger: Optional logger
    
    Returns:
        Metrics dictionary
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    
    # Track metrics
    metrics_tracker.add_result(
        model_name=model_name,
        fold=None,
        **metrics,
        evaluation_type="test",
    )
    
    if logger:
        logger.info(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return metrics


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions and submissions from trained models",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to use: specific model name, 'all' for all models, or 'ensemble'",
    )
    
    parser.add_argument(
        "--ensemble-method",
        type=str,
        choices=["mean", "weighted"],
        default="mean",
        help="Ensemble method (only used with --model ensemble)",
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for weighted ensemble (in order: catboost, lightgbm, xgboost, mlp)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output filename (without path, without .csv)",
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup paths
    data_dir = get_data_dir()
    output_dir = get_output_dir()
    models_dir = output_dir / "models"
    
    # Setup logger
    logger = get_logger("evaluate", log_dir=output_dir / "logs")
    
    logger.info("=" * 60)
    logger.info("EVALUATION SCRIPT")
    logger.info("=" * 60)
    
    # Load test data
    logger.info("Loading test data...")
    test_df, sample_df = load_test_data(data_dir)
    logger.info(f"Test data: {len(test_df):,} rows")
    
    # Store IDs before preprocessing drops them
    test_ids = test_df[ID_COLUMN].copy()
    
    # Check for ground truth (unlikely in test data, but handle anyway)
    has_ground_truth = TARGET_COLUMN in test_df.columns
    if has_ground_truth:
        y_true = test_df[TARGET_COLUMN].values
        test_df = test_df.drop(columns=[TARGET_COLUMN])
        logger.info("Ground truth found in test data - will calculate metrics")
        metrics_tracker = MetricsTracker(output_dir=output_dir / "metrics")
    
    # Get available models
    available_models = get_available_models(models_dir)
    logger.info(f"Available trained models: {available_models}")
    
    if not available_models:
        logger.error("No trained models found! Run training first.")
        return
    
    # Determine which models to use
    if args.model == "all":
        models_to_use = available_models
        mode = "all"
    elif args.model == "ensemble":
        models_to_use = available_models
        mode = "ensemble"
    elif args.model in MODELS:
        if args.model not in available_models:
            logger.error(f"Model '{args.model}' not found in {models_dir}")
            return
        models_to_use = [args.model]
        mode = "single"
    else:
        logger.error(f"Unknown model: {args.model}")
        return
    
    logger.info(f"Mode: {mode}, Models: {models_to_use}")
    
    # Generate predictions
    if mode == "ensemble":
        # Ensemble prediction
        predictions = ensemble_predict(
            model_names=models_to_use,
            test_df=test_df,
            models_dir=models_dir,
            method=args.ensemble_method,
            weights=args.weights,
            logger=logger,
        )
        
        model_label = f"ensemble_{args.ensemble_method}"
        if args.output:
            model_label = args.output
        
        filepath = generate_submission(
            ids=test_ids,
            predictions=predictions,
            output_dir=output_dir,
            model_name=model_label,
            logger=logger,
        )
        
        if has_ground_truth:
            calculate_metrics(y_true, predictions, model_label, metrics_tracker, logger)
    
    else:
        # Single or all models
        for model_name in models_to_use:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing: {model_name}")
            logger.info(f"{'='*40}")
            
            predictions = predict_single_model(
                model_name=model_name,
                test_df=test_df,
                models_dir=models_dir,
                logger=logger,
            )
            
            model_label = model_name
            if args.output and len(models_to_use) == 1:
                model_label = args.output
            
            filepath = generate_submission(
                ids=test_ids,
                predictions=predictions,
                output_dir=output_dir,
                model_name=model_label,
                logger=logger,
            )
            
            if has_ground_truth:
                calculate_metrics(y_true, predictions, model_name, metrics_tracker, logger)
    
    # Export metrics if ground truth was available
    if has_ground_truth:
        metrics_tracker.to_excel("test_metrics.xlsx")
        metrics_tracker.to_json("test_metrics.json")
        logger.info("Metrics exported to outputs/metrics/")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
