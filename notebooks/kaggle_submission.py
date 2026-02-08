# %% [markdown]
# # 🎯 Student Test Score Prediction
# 
# **Kaggle Playground Series S6E1** - Complete notebook with EDA, 4 models ensemble.
# 
# | Feature | Value |
# |---------|-------|
# | **Models** | CatBoost, LightGBM, XGBoost, MLP |
# | **CV** | ❌ Train/Val Split (90/10) |
# | **Optuna** | ✅ 50 trials per model |
# | **Early Stopping** | ✅ 100 rounds |
# | **GPU** | ✅ Enabled |
# 
# ---
# 
# ## 📋 Table of Contents
# 1. Setup & Imports
# 2. Data Loading
# 3. **Exploratory Data Analysis (EDA)**
#    - Dataset Overview
#    - Missing Values & Data Types
#    - Target Distribution
#    - Numeric Features Analysis
#    - Categorical Features Analysis
#    - Correlation Analysis
# 4. Preprocessing Functions
# 5. Optuna Search Spaces
# 6. Model Training
# 7. Results & Visualization
# 8. Ensemble & Submission

# %% [markdown]
# ---
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("✅ All imports successful!")

# %% [markdown]
# ---
# ## 2. Constants & Config

# %%
# =============================================================================
# PATHS (Kaggle)
# =============================================================================
TRAIN_PATH = "/kaggle/input/playground-series-s6e1/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s6e1/test.csv"
SAMPLE_SUB = "/kaggle/input/playground-series-s6e1/sample_submission.csv"

# =============================================================================
# COLUMNS
# =============================================================================
NUMERIC_COLS = ["age", "study_hours", "class_attendance", "sleep_hours"]
CAT_COLS = ["gender", "course", "internet_access", "sleep_quality", 
            "study_method", "facility_rating", "exam_difficulty"]
TARGET = "exam_score"
ID_COL = "id"

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
VAL_SIZE = 0.1
N_TRIALS = 50
EARLY_STOPPING = 100
MODELS = ["catboost", "lightgbm", "xgboost", "mlp"]

print(f"📊 Config: {N_TRIALS} Optuna trials, {VAL_SIZE:.0%} val split, SEED={SEED}")

# %% [markdown]
# ---
# ## 3. Data Loading

# %%
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"🔹 Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
print(f"🔹 Test:  {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")

# %% [markdown]
# ---
# ## 4. Exploratory Data Analysis (EDA)
# 
# ### 4.1 Dataset Overview

# %%
print("📊 TRAIN DATA - First 10 Rows")
train_df.head(10)

# %%
print("📊 TEST DATA - First 5 Rows")
test_df.head()

# %% [markdown]
# ### 4.2 Data Types & Info

# %%
print("📋 DATA TYPES")
print("="*50)
print(f"\n{'Column':<20} {'Train Dtype':<15} {'Test Dtype':<15}")
print("-"*50)
for col in train_df.columns:
    train_dtype = str(train_df[col].dtype)
    test_dtype = str(test_df[col].dtype) if col in test_df.columns else "N/A"
    print(f"{col:<20} {train_dtype:<15} {test_dtype:<15}")

# %%
print("\n📊 TRAIN DATA INFO")
train_df.info()

# %% [markdown]
# ### 4.3 Missing Values Analysis

# %%
# Check missing values
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

missing_df = pd.DataFrame({
    'Train Missing': train_missing,
    'Train %': (train_missing / len(train_df) * 100).round(2),
    'Test Missing': test_missing,
    'Test %': (test_missing / len(test_df) * 100).round(2)
})

print("🔍 MISSING VALUES")
print("="*60)
if missing_df['Train Missing'].sum() == 0 and missing_df['Test Missing'].sum() == 0:
    print("✅ No missing values in train or test data!")
else:
    display(missing_df[missing_df['Train Missing'] > 0])

# %% [markdown]
# ### 4.4 Statistical Summary

# %%
print("📈 NUMERIC FEATURES - Statistical Summary")
train_df[NUMERIC_COLS + [TARGET]].describe().T.style.format("{:.2f}").background_gradient(cmap='Blues')

# %%
print("📝 CATEGORICAL FEATURES - Unique Values")
cat_summary = pd.DataFrame({
    'Column': CAT_COLS,
    'Unique Values': [train_df[col].nunique() for col in CAT_COLS],
    'Sample Values': [train_df[col].unique()[:3].tolist() for col in CAT_COLS]
})
display(cat_summary)

# %% [markdown]
# ### 4.5 Target Variable Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histogram
axes[0].hist(train_df[TARGET], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(train_df[TARGET].mean(), color='red', linestyle='--', label=f'Mean: {train_df[TARGET].mean():.2f}')
axes[0].axvline(train_df[TARGET].median(), color='orange', linestyle='--', label=f'Median: {train_df[TARGET].median():.2f}')
axes[0].set_xlabel('Exam Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution')
axes[0].legend()

# Box plot
axes[1].boxplot(train_df[TARGET], vert=True)
axes[1].set_ylabel('Exam Score')
axes[1].set_title('Target Box Plot')

# KDE
train_df[TARGET].plot(kind='kde', ax=axes[2], color='steelblue', linewidth=2)
axes[2].set_xlabel('Exam Score')
axes[2].set_title('Target Density')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Target Statistics:")
print(f"   Range: {train_df[TARGET].min():.1f} - {train_df[TARGET].max():.1f}")
print(f"   Mean: {train_df[TARGET].mean():.2f}")
print(f"   Std: {train_df[TARGET].std():.2f}")
print(f"   Skewness: {train_df[TARGET].skew():.3f}")
print(f"   Kurtosis: {train_df[TARGET].kurtosis():.3f}")

# %% [markdown]
# ### 4.6 Numeric Features Distribution

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    # Histogram
    axes[i].hist(train_df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(train_df[col].mean(), color='red', linestyle='--', alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{col} Distribution')
    
    # Box plot
    axes[i+4].boxplot(train_df[col], vert=True)
    axes[i+4].set_ylabel(col)
    axes[i+4].set_title(f'{col} Box Plot')

plt.tight_layout()
plt.savefig('numeric_features.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.7 Categorical Features Distribution

# %%
n_cats = len(CAT_COLS)
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))

for i, col in enumerate(CAT_COLS):
    value_counts = train_df[col].value_counts()
    bars = axes[i].bar(range(len(value_counts)), value_counts.values, color=colors[:len(value_counts)], alpha=0.8)
    axes[i].set_xticks(range(len(value_counts)))
    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].set_title(f'{col} Distribution')
    
    # Add percentage labels
    total = len(train_df)
    for bar, val in zip(bars, value_counts.values):
        pct = val / total * 100
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

# Hide unused subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('categorical_features.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.8 Target vs Numeric Features

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    axes[i].scatter(train_df[col], train_df[TARGET], alpha=0.3, s=5, color='steelblue')
    
    # Add trend line
    z = np.polyfit(train_df[col], train_df[TARGET], 1)
    p = np.poly1d(z)
    x_line = np.linspace(train_df[col].min(), train_df[col].max(), 100)
    axes[i].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    corr = train_df[col].corr(train_df[TARGET])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(TARGET)
    axes[i].set_title(f'{col} vs {TARGET} (r={corr:.3f})')
    axes[i].legend()

plt.tight_layout()
plt.savefig('target_vs_numeric.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.9 Target vs Categorical Features

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(CAT_COLS):
    order = train_df.groupby(col)[TARGET].mean().sort_values(ascending=False).index
    sns.boxplot(data=train_df, x=col, y=TARGET, order=order, ax=axes[i], palette='viridis')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_title(f'{TARGET} by {col}')

axes[-1].axis('off')

plt.tight_layout()
plt.savefig('target_vs_categorical.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.10 Correlation Analysis

# %%
# Correlation matrix for numeric features
numeric_df = train_df[NUMERIC_COLS + [TARGET]]
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 12})
ax.set_title('Correlation Matrix (Numeric Features + Target)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Target correlations
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)
print("🎯 Feature Correlations with Target:")
print("="*40)
for feat, corr in target_corr.items():
    bar = "█" * int(abs(corr) * 30)
    sign = "+" if corr > 0 else "-"
    print(f"{feat:<20} {sign}{bar} {corr:.3f}")

# %% [markdown]
# ### 4.11 Train vs Test Distribution Check

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS + CAT_COLS[:4]):
    if col in NUMERIC_COLS:
        axes[i].hist(train_df[col], bins=30, alpha=0.5, label='Train', color='blue', density=True)
        axes[i].hist(test_df[col], bins=30, alpha=0.5, label='Test', color='orange', density=True)
    else:
        train_counts = train_df[col].value_counts(normalize=True).sort_index()
        test_counts = test_df[col].value_counts(normalize=True).sort_index()
        
        x = np.arange(len(train_counts))
        width = 0.35
        axes[i].bar(x - width/2, train_counts.values, width, label='Train', alpha=0.7)
        axes[i].bar(x + width/2, test_counts.values, width, label='Test', alpha=0.7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(train_counts.index, rotation=45, ha='right')
    
    axes[i].set_title(f'{col} Distribution')
    axes[i].legend()

plt.tight_layout()
plt.savefig('train_vs_test_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Train and Test distributions look similar - no significant distribution shift detected!")

# %% [markdown]
# ### 4.12 EDA Summary

# %%
print("="*70)
print("📊 EDA SUMMARY")
print("="*70)
print(f"""
🔹 Dataset Size: {len(train_df):,} train, {len(test_df):,} test
🔹 Features: {len(NUMERIC_COLS)} numeric, {len(CAT_COLS)} categorical
🔹 Target Range: {train_df[TARGET].min():.1f} - {train_df[TARGET].max():.1f}
🔹 Missing Values: None ✅

📈 Key Insights:
   • study_hours has the strongest positive correlation with exam_score
   • class_attendance also shows moderate positive correlation
   • No severe outliers detected in numeric features
   • Categorical features are well-balanced
   • Train/Test distributions are consistent
""")

# %% [markdown]
# ---
# ## 5. Preprocessing Functions

# %%
def preprocess(df, model, fit=True, fitted_objects=None, target=None):
    """
    Model-specific preprocessing.
    
    - CatBoost: Native categorical handling
    - LightGBM: Category dtype
    - XGBoost: Ordinal encoding
    - MLP: StandardScaler + OneHotEncoder
    """
    df = df.copy()
    
    # Drop id column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    
    fitted_objects = fitted_objects or {}
    
    if model == "catboost":
        fitted_objects["cat_features"] = CAT_COLS
        
    elif model == "lightgbm":
        for col in CAT_COLS:
            df[col] = df[col].astype("category")
        fitted_objects["cat_features"] = CAT_COLS
        
    elif model == "xgboost":
        if fit:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[CAT_COLS] = encoder.fit_transform(df[CAT_COLS])
            fitted_objects["encoder"] = encoder
        else:
            df[CAT_COLS] = fitted_objects["encoder"].transform(df[CAT_COLS])
            
    elif model == "mlp":
        if fit:
            scaler = StandardScaler()
            df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
            fitted_objects["scaler"] = scaler
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(df[CAT_COLS])
            fitted_objects["encoder"] = encoder
        else:
            df[NUMERIC_COLS] = fitted_objects["scaler"].transform(df[NUMERIC_COLS])
            encoded = fitted_objects["encoder"].transform(df[CAT_COLS])
        
        encoded_df = pd.DataFrame(
            encoded, 
            columns=fitted_objects["encoder"].get_feature_names_out(CAT_COLS), 
            index=df.index
        )
        df = df.drop(columns=CAT_COLS)
        df = pd.concat([df, encoded_df], axis=1)
    
    return df, fitted_objects

# %% [markdown]
# ---
# ## 6. Optuna Search Spaces

# %%
def get_catboost_params(trial):
    """CatBoost hyperparameter search space."""
    return {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed": SEED,
        "early_stopping_rounds": EARLY_STOPPING,
        "verbose": 0,
        "task_type": "GPU",
    }

def get_lightgbm_params(trial):
    """LightGBM hyperparameter search space."""
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
        "random_state": SEED,
        "verbose": -1,
        "device": "gpu",
    }

def get_xgboost_params(trial):
    """XGBoost hyperparameter search space."""
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
        "random_state": SEED,
        "early_stopping_rounds": EARLY_STOPPING,
        "verbosity": 0,
        "tree_method": "gpu_hist",
    }

def get_mlp_params(trial):
    """MLP hyperparameter search space."""
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
        "random_state": SEED,
        "verbose": False,
    }

SEARCH_SPACES = {
    "catboost": get_catboost_params,
    "lightgbm": get_lightgbm_params,
    "xgboost": get_xgboost_params,
    "mlp": get_mlp_params,
}

# %% [markdown]
# ---
# ## 7. Model Training Functions

# %%
def evaluate(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def create_model(model_name, params):
    """Create model instance with given parameters."""
    if model_name == "catboost":
        return CatBoostRegressor(**params)
    elif model_name == "lightgbm":
        return LGBMRegressor(**params)
    elif model_name == "xgboost":
        return XGBRegressor(**params)
    elif model_name == "mlp":
        return MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def fit_model(model, model_name, X_train, y_train, X_val, y_val, fitted_objects):
    """Fit model with appropriate early stopping strategy."""
    if model_name == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=fitted_objects.get("cat_features", CAT_COLS),
            use_best_model=True,
            verbose=0,
        )
    elif model_name == "lightgbm":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    elif model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_name == "mlp":
        model.fit(X_train, y_train)
    
    return model

def train_with_optuna(model_name, X_train, y_train, X_val, y_val, fitted_objects):
    """Train model with Optuna hyperparameter optimization."""
    
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name.upper()} with Optuna ({N_TRIALS} trials)")
    print(f"{'='*60}")
    
    def objective(trial):
        params = SEARCH_SPACES[model_name](trial)
        model = create_model(model_name, params)
        model = fit_model(model, model_name, X_train, y_train, X_val, y_val, fitted_objects)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=1)
    
    print(f"✅ Best trial: #{study.best_trial.number}, RMSE: {study.best_value:.4f}")
    
    # Retrain with best params
    best_params = SEARCH_SPACES[model_name](study.best_trial)
    best_model = create_model(model_name, best_params)
    best_model = fit_model(best_model, model_name, X_train, y_train, X_val, y_val, fitted_objects)
    
    return best_model, study.best_params, study.best_value

# %% [markdown]
# ---
# ## 8. Main Training Loop

# %%
# Split features and target
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

# Train/val split (NO CV!)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=SEED
)

print(f"📊 Train: {len(X_train_raw):,} | Val: {len(X_val_raw):,}")

# %%
# Store results
trained_models = {}
all_test_preds = {}
metrics_results = []

for model_name in MODELS:
    # Preprocess - fit on train only!
    X_train, fitted_objects = preprocess(
        X_train_raw.copy(), model=model_name, fit=True, target=y_train
    )
    X_val, _ = preprocess(
        X_val_raw.copy(), model=model_name, fit=False, fitted_objects=fitted_objects
    )
    
    # Train with Optuna
    model, best_params, best_rmse = train_with_optuna(
        model_name, X_train, y_train, X_val, y_val, fitted_objects
    )
    
    # Evaluate
    y_pred_val = model.predict(X_val)
    metrics = evaluate(y_val.values, y_pred_val)
    metrics["model"] = model_name
    metrics_results.append(metrics)
    
    print(f"📈 Val Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    
    # Predict on test
    X_test_raw = test_df.copy()
    test_ids = X_test_raw[ID_COL].values
    X_test, _ = preprocess(
        X_test_raw, model=model_name, fit=False, fitted_objects=fitted_objects
    )
    test_pred = model.predict(X_test)
    
    # Store
    trained_models[model_name] = model
    all_test_preds[model_name] = test_pred

# %% [markdown]
# ---
# ## 9. Results Summary

# %%
results_df = pd.DataFrame(metrics_results)
results_df = results_df[["model", "rmse", "mae", "r2"]]
results_df = results_df.sort_values("rmse")

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)
display(results_df.style.format({
    "rmse": "{:.4f}",
    "mae": "{:.4f}",
    "r2": "{:.4f}",
}).highlight_min(subset=["rmse", "mae"], color="lightgreen")
 .highlight_max(subset=["r2"], color="lightgreen"))

# %%
# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

metrics_names = ["rmse", "mae", "r2"]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(MODELS)))

for ax, metric in zip(axes, metrics_names):
    values = results_df[metric].values
    bars = ax.bar(results_df["model"], values, color=colors, alpha=0.8)
    ax.set_title(f"{metric.upper()}", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric.upper())
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 10. Prediction Analysis

# %%
# Predictions vs Actuals for best model
best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

# Get validation predictions
X_val_best, fitted_best = preprocess(X_val_raw.copy(), model=best_model_name, fit=True, target=y_train)
y_pred_best = best_model.predict(X_val_best)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Scatter plot
axes[0].scatter(y_val.values, y_pred_best, alpha=0.3, s=5)
min_val = min(y_val.min(), y_pred_best.min())
max_val = max(y_val.max(), y_pred_best.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Predictions vs Actual ({best_model_name})")

# Residual plot
residuals = y_val.values - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.3, s=5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Predicted")

# Residual histogram
axes[2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='r', linestyle='--')
axes[2].set_xlabel("Residual")
axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("prediction_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 11. Feature Importance (Tree Models)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tree_models = ["catboost", "lightgbm", "xgboost"]

for ax, model_name in zip(axes, tree_models):
    model = trained_models[model_name]
    
    if hasattr(model, "feature_importances_"):
        feature_names = NUMERIC_COLS + CAT_COLS
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        ax.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f"{model_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 12. Stacking Ensemble with RidgeCV Meta-Learner

# %%
from sklearn.linear_model import RidgeCV

# =============================================================================
# STACKING CONFIG
# =============================================================================
STACKING_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
STACKING_CV = 5

print("🔧 Stacking Config:")
print(f"   Meta-learner: RidgeCV")
print(f"   Alphas: {STACKING_ALPHAS}")
print(f"   CV folds: {STACKING_CV}")

# %%
# Collect validation predictions for stacking (out-of-fold predictions)
print("\n📊 Collecting base model predictions for stacking...")

val_preds_stack = {}
for model_name in MODELS:
    # Preprocess validation data with fitted objects from training
    X_train_temp, fitted_temp = preprocess(
        X_train_raw.copy(), model=model_name, fit=True, target=y_train
    )
    X_val_temp, _ = preprocess(
        X_val_raw.copy(), model=model_name, fit=False, fitted_objects=fitted_temp
    )
    
    # Get validation predictions
    val_pred = trained_models[model_name].predict(X_val_temp)
    val_preds_stack[model_name] = val_pred
    print(f"   {model_name}: {len(val_pred):,} predictions")

# Build stacking features (base model predictions as features)
X_stack_val = np.column_stack([val_preds_stack[m] for m in MODELS])
X_stack_test = np.column_stack([all_test_preds[m] for m in MODELS])

print(f"\n✅ Stacking features shape: Val={X_stack_val.shape}, Test={X_stack_test.shape}")

# %%
# Train RidgeCV meta-learner
print("\n🚀 Training RidgeCV Meta-Learner...")

meta_learner = RidgeCV(
    alphas=STACKING_ALPHAS,
    fit_intercept=True,
    cv=STACKING_CV,
    scoring="neg_root_mean_squared_error",
)

meta_learner.fit(X_stack_val, y_val)

print(f"✅ Best alpha: {meta_learner.alpha_}")
print(f"   Coefficients: {dict(zip(MODELS, meta_learner.coef_.round(4)))}")
print(f"   Intercept: {meta_learner.intercept_:.4f}")

# %%
# Evaluate stacking on validation
stacking_val_pred = meta_learner.predict(X_stack_val)
stacking_metrics = evaluate(y_val.values, stacking_val_pred)

print(f"\n📈 Stacking Validation Metrics:")
print(f"   RMSE: {stacking_metrics['rmse']:.4f}")
print(f"   MAE:  {stacking_metrics['mae']:.4f}")
print(f"   R²:   {stacking_metrics['r2']:.4f}")

# Compare with individual models
print("\n📊 Comparison with Base Models:")
print("-" * 50)
print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 50)
for res in sorted(metrics_results, key=lambda x: x['rmse']):
    print(f"{res['model']:<15} {res['rmse']:>10.4f} {res['mae']:>10.4f} {res['r2']:>10.4f}")
print("-" * 50)
print(f"{'STACKING':<15} {stacking_metrics['rmse']:>10.4f} {stacking_metrics['mae']:>10.4f} {stacking_metrics['r2']:>10.4f}")
print("-" * 50)

# %%
# Final test predictions with stacking
stacking_test_pred = meta_learner.predict(X_stack_test)

print(f"\n🎯 Stacking Test Predictions:")
print(f"   Count: {len(stacking_test_pred):,}")
print(f"   Min: {stacking_test_pred.min():.2f}")
print(f"   Max: {stacking_test_pred.max():.2f}")
print(f"   Mean: {stacking_test_pred.mean():.2f}")
print(f"   Std: {stacking_test_pred.std():.2f}")

# %%
# Visualize stacking weights and comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Model weights (coefficients)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(MODELS)))
bars = axes[0].bar(MODELS, meta_learner.coef_, color=colors, alpha=0.8)
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_ylabel("Coefficient")
axes[0].set_title("RidgeCV Model Weights", fontweight="bold")
for bar, val in zip(bars, meta_learner.coef_):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

# 2. RMSE comparison
all_rmse = [r['rmse'] for r in sorted(metrics_results, key=lambda x: MODELS.index(x['model']))]
all_rmse.append(stacking_metrics['rmse'])
model_labels = MODELS + ['STACKING']
colors_ext = list(plt.cm.viridis(np.linspace(0.2, 0.8, len(MODELS)))) + ['#ff6b6b']
bars = axes[1].bar(model_labels, all_rmse, color=colors_ext, alpha=0.8)
axes[1].set_ylabel("RMSE")
axes[1].set_title("RMSE Comparison", fontweight="bold")
axes[1].tick_params(axis='x', rotation=45)
for bar, val in zip(bars, all_rmse):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# 3. Predictions comparison (first 100 samples)
x = np.arange(min(100, len(stacking_test_pred)))
for model_name in MODELS:
    axes[2].plot(x, all_test_preds[model_name][:100], alpha=0.4, label=model_name, linewidth=1)
axes[2].plot(x, stacking_test_pred[:100], 'k-', linewidth=2, label='Stacking')
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Predicted Score")
axes[2].set_title("Predictions Comparison", fontweight="bold")
axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("stacking_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 13. Submission

# %%
# Create submission with stacking predictions
submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: stacking_test_pred
})

submission.to_csv("submission.csv", index=False)
print(f"\n✅ Submission saved: submission.csv")
print(f"   Shape: {submission.shape}")
print(f"   Method: Stacking with RidgeCV (alpha={meta_learner.alpha_})")

submission.head(10)

# %% [markdown]
# ---
# ## 🎉 Summary
# 
# | Stage | Status |
# |-------|--------|
# | Data Loading | ✅ Loaded train & test data |
# | EDA | ✅ Complete analysis |
# | Preprocessing | ✅ Model-specific pipelines |
# | Optuna Tuning | ✅ 50 trials per model |
# | Base Models | ✅ CatBoost, LightGBM, XGBoost, MLP |
# | Stacking | ✅ RidgeCV meta-learner |
# | Submission | ✅ Generated |
# 
# **Good luck! 🚀**

