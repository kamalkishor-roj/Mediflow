import polars as pl
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "data/raw/sepsis_data.csv"
TARGET_COL = "SepsisLabel"
RANDOM_STATE = 42

# ---------------------------
# Load data
# ---------------------------
df = pl.read_csv(DATA_PATH)

# Drop rows with null target (critical)
df = df.filter(pl.col(TARGET_COL).is_not_null())

# ---------------------------
# Split features / target
# ---------------------------
X = df.drop(TARGET_COL)
y = df[TARGET_COL].to_numpy()

# Convert to NumPy (XGBoost requirement)
X_np = X.to_numpy()

# ---------------------------
# Train / Validation split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_np,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------------------------
# Handle class imbalance
# ---------------------------
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---------------------------
# XGBoost parameters
# ---------------------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": pos_weight,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

# ---------------------------
# Train with MLflow tracking
# ---------------------------
with mlflow.start_run():
    model = xgb.XGBClassifier(
        **params,
        n_estimators=300
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_val_pred = model.predict_proba(X_val)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_val, y_val_pred)

    precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
    pr_auc = auc(recall, precision)

    # Log to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    mlflow.xgboost.log_model(model, "sepsis_risk_model")

print("✅ Training completed successfully")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
