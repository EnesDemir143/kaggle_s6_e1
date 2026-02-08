import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    TargetEncoder,
)

# Column definitions
NUMERIC_COLUMNS = ["age", "study_hours", "class_attendance", "sleep_hours"]
CATEGORICAL_COLUMNS = [
    "gender", "course", "internet_access", "sleep_quality",
    "study_method", "facility_rating", "exam_difficulty",
]
TARGET_COLUMN = "exam_score"
ID_COLUMN = "id"

NormMethod = Literal["standard", "minmax", "robust"]
EncMethod = Literal["onehot", "label", "ordinal", "target"]
ModelType = Literal["catboost", "lightgbm", "xgboost", "mlp"]


# =============================================================================
# NORMALIZER
# =============================================================================

def get_normalizer(method: NormMethod = "standard"):
    """Get scaler: standard, minmax, or robust."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unknown method: {method}")
    return scalers[method]()


def fit_transform_normalizer(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: NormMethod = "standard",
    scaler=None,
    fit: bool = True,
) -> tuple[pd.DataFrame, object]:
    """Fit and/or transform normalizer."""
    columns = columns or NUMERIC_COLUMNS
    df_copy = df.copy()
    
    if fit:
        scaler = get_normalizer(method)
        scaler.fit(df_copy[columns])
    
    df_copy[columns] = scaler.transform(df[columns])
    return df_copy, scaler


def get_normalization_stats(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Get mean, std, min, max for numeric columns."""
    columns = columns or NUMERIC_COLUMNS
    numeric_df = df[columns]
    return {
        "columns": np.array(columns),
        "mean": numeric_df.mean().values.astype(np.float32),
        "std": numeric_df.std().values.astype(np.float32),
        "min": numeric_df.min().values.astype(np.float32),
        "max": numeric_df.max().values.astype(np.float32),
    }


# =============================================================================
# ENCODER
# =============================================================================

def fit_transform_encoder(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: EncMethod = "onehot",
    encoder=None,
    fit: bool = True,
    target: pd.Series | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Fit and/or transform encoder.
    
    Methods:
        - onehot: OneHotEncoder (for MLP, linear models)
        - label: LabelEncoder (simple integer)
        - ordinal: OrdinalEncoder (integer for tree models)
        - target: TargetEncoder (based on target mean)
    """
    columns = columns or CATEGORICAL_COLUMNS
    df_copy = df.copy()
    
    if fit:
        if method == "onehot":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(df_copy[columns])
        elif method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoder.fit(df_copy[columns])
        elif method == "label":
            encoder = {col: LabelEncoder().fit(df_copy[col]) for col in columns}
        elif method == "target":
            if target is None:
                raise ValueError("Target required for target encoding")
            encoder = TargetEncoder()
            encoder.fit(df_copy[columns], target)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Transform
    if method == "onehot":
        encoded = encoder.transform(df_copy[columns])
        encoded_df = pd.DataFrame(
            encoded, columns=encoder.get_feature_names_out(columns), index=df.index
        )
        df_copy = df_copy.drop(columns=columns)
        df_copy = pd.concat([df_copy, encoded_df], axis=1)
    elif method == "ordinal":
        df_copy[columns] = encoder.transform(df_copy[columns])
    elif method == "label":
        for col in columns:
            df_copy[col] = encoder[col].transform(df_copy[col])
    elif method == "target":
        df_copy[columns] = encoder.transform(df_copy[columns])
    
    return df_copy, encoder


# =============================================================================
# MODEL-SPECIFIC PREPROCESSING
# =============================================================================

def preprocess(
    df: pd.DataFrame,
    model: ModelType,
    fit: bool = True,
    fitted_objects: dict | None = None,
    norm_method: NormMethod = "standard",
    enc_method: EncMethod = "ordinal",
    target: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocess data for specific model.
    
    Args:
        df: Input DataFrame
        model: Model type (catboost, lightgbm, xgboost, mlp)
        fit: True for training, False for inference
        fitted_objects: Pre-fitted objects (for inference)
        norm_method: Normalization method (for MLP)
        enc_method: Encoding method (for XGBoost, MLP)
        target: Target variable (for target encoding)
    
    Returns:
        (preprocessed_df, fitted_objects)
    """
    df_copy = df.copy()
    
    # Drop id
    if ID_COLUMN in df_copy.columns:
        df_copy = df_copy.drop(columns=[ID_COLUMN])
    
    fitted_objects = fitted_objects or {}
    
    match model:
        case "catboost":
            # CatBoost handles categorical natively
            fitted_objects["cat_features"] = CATEGORICAL_COLUMNS
            
        case "lightgbm":
            # LightGBM uses category dtype
            for col in CATEGORICAL_COLUMNS:
                df_copy[col] = df_copy[col].astype("category")
            fitted_objects["cat_features"] = CATEGORICAL_COLUMNS
            
        case "xgboost":
            # XGBoost needs encoding
            df_copy, fitted_objects["encoder"] = fit_transform_encoder(
                df_copy, CATEGORICAL_COLUMNS, enc_method,
                encoder=fitted_objects.get("encoder"),
                fit=fit, target=target,
            )
            fitted_objects["enc_method"] = enc_method
            
        case "mlp":
            # MLP needs normalization + encoding
            df_copy, fitted_objects["scaler"] = fit_transform_normalizer(
                df_copy, NUMERIC_COLUMNS, norm_method,
                scaler=fitted_objects.get("scaler"),
                fit=fit,
            )
            # Default onehot for MLP
            enc = enc_method if enc_method != "ordinal" else "onehot"
            df_copy, fitted_objects["encoder"] = fit_transform_encoder(
                df_copy, CATEGORICAL_COLUMNS, enc,
                encoder=fitted_objects.get("encoder"),
                fit=fit, target=target,
            )
            fitted_objects["norm_method"] = norm_method
            fitted_objects["enc_method"] = enc
            
        case _:
            raise ValueError(f"Unknown model: {model}")
    
    return df_copy, fitted_objects


# =============================================================================
# UTILITIES
# =============================================================================

def load_data(
    train_path: str | Path,
    test_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train and optionally test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split DataFrame into features and target."""
    if target_col in df.columns:
        return df.drop(columns=[target_col]), df[target_col]
    return df, None


def save_fitted_objects(fitted_objects: dict, output_path: str | Path) -> None:
    """Save fitted preprocessing objects."""
    import joblib
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_objects, output_path)


def load_fitted_objects(input_path: str | Path) -> dict:
    """Load fitted preprocessing objects."""
    import joblib
    return joblib.load(input_path)
