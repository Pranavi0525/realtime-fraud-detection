"""
IEEE-CIS Fraud Detection Model Trainer
=======================================
Trains an XGBoost model on the IEEE-CIS Kaggle dataset.

Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data
Required files (place in /data/ieee/):
  - train_transaction.csv
  - train_identity.csv  (optional — improves model)

Run:
  python scripts/train_ieee_model.py

Outputs to /models/:
  - ieee_fraud_model.joblib
  - ieee_scaler.joblib
  - ieee_feature_names.json
  - ieee_model_metadata.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/ieee")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Features we care about ────────────────────────────────────────────────────
# These are the transaction-level signals that matter for real-time fraud detection
# No identity features — we want inference to work from transaction data alone.

TRANSACTION_FEATURES = [
    "TransactionAmt",         # Amount — core signal
    "ProductCD",              # Product code (W, H, C, S, R)
    "card1", "card2", "card3", "card5",  # Card attributes
    "addr1", "addr2",         # Billing address codes
    "dist1",                  # Distance from home
    "P_emaildomain",          # Purchaser email domain
    "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12", "C13", "C14",  # Counting features
    "D1", "D2", "D3", "D4", "D10", "D15",  # Timedelta features
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",  # Match features
    "V12", "V13", "V14", "V17", "V20", "V23", "V26", "V27",
    "V29", "V30", "V35", "V36", "V37", "V38", "V45", "V46",
    "V47", "V48", "V49", "V53", "V54", "V56", "V57", "V58",
    "V59", "V60", "V61", "V62", "V63", "V64", "V69", "V70",
    "V75", "V76", "V78", "V80", "V81", "V82", "V83", "V84",
    "V85", "V86", "V87", "V90", "V91", "V92", "V93", "V94",
    "V95", "V96", "V97", "V98", "V99", "V100",
    "TransactionDT",          # Time delta from reference date
]

CATEGORICAL_FEATURES = [
    "ProductCD", "P_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


def load_data():
    print("Loading IEEE-CIS transaction data...")
    txn_path = DATA_DIR / "train_transaction.csv"
    if not txn_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {txn_path}\n"
            "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
            "Place train_transaction.csv in data/ieee/"
        )
    df = pd.read_csv(txn_path)
    print(f"  Loaded {len(df):,} transactions | Fraud rate: {df['isFraud'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering on IEEE-CIS data for real-time fraud detection."""
    print("Engineering features...")

    # Log-transform amount — heavy right skew
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Hour of day (TransactionDT is seconds from reference)
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24

    # Is weekend (day of week)
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Amount buckets (round amounts are more common in fraud)
    df["amt_is_round"] = (df["TransactionAmt"] % 10 == 0).astype(int)

    # High-value transaction flag
    df["is_high_value"] = (df["TransactionAmt"] > 500).astype(int)

    return df


def preprocess(df: pd.DataFrame):
    """Select features, encode categoricals, fill nulls."""
    # Add engineered features to our list
    engineered = ["TransactionAmt_log", "hour_of_day", "day_of_week", "is_weekend",
                  "amt_is_round", "is_high_value"]

    all_features = [f for f in TRANSACTION_FEATURES if f in df.columns] + engineered

    X = df[all_features].copy()

    # Label-encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    # Fill nulls with -999 (XGBoost handles this well)
    X = X.fillna(-999)

    y = df["isFraud"]
    return X, y, all_features


def train(X_train, y_train, X_val, y_val):
    """Train XGBoost with scale_pos_weight to handle class imbalance."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"  Class imbalance: {neg:,} legit | {pos:,} fraud | scale_pos_weight={spw:.1f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate(model, X_test, y_test, feature_names):
    """Evaluate and print metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    prauc = average_precision_score(y_test, proba)

    # Find threshold maximising F1
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, proba >= t) for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1s)])

    print(f"\n{'='*50}")
    print(f"  AUC-ROC        : {auc:.4f}")
    print(f"  PR-AUC         : {prauc:.4f}")
    print(f"  Best Threshold : {best_threshold:.4f}")
    print(f"{'='*50}")

    preds = (proba >= best_threshold).astype(int)
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

    # Top 10 features by importance
    importance = dict(zip(feature_names, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features:")
    for name, imp in top10:
        print(f"  {name:<30} {imp:.4f}")

    return auc, prauc, best_threshold, top10


def save_artifacts(model, scaler, feature_names, auc, prauc, threshold, top_features):
    """Save model + metadata for fraud detection service."""
    joblib.dump(model,  MODELS_DIR / "ieee_fraud_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "ieee_scaler.joblib")

    with open(MODELS_DIR / "ieee_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metadata = {
        "model_type": "XGBoost — IEEE-CIS Transaction Fraud",
        "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
        "auc_roc": round(auc, 4),
        "pr_auc": round(prauc, 4),
        "optimal_threshold": round(threshold, 4),
        "top_features": [{"feature": k, "importance": round(v, 4)} for k, v in top_features],
        "training_note": (
            "Trained on real IEEE-CIS transaction data. "
            "Features are transaction-level signals — amount, timing, card attributes, "
            "counting features (C1-C14), timedelta features (D1-D15), and Vesta-engineered "
            "features (V12-V100). Directly applicable to card transaction fraud detection."
        ),
        "feature_note": (
            "At inference time, pass the same features. "
            "Missing features are filled with -999. "
            "Categoricals (ProductCD, email domain, M-features) are label-encoded."
        )
    }
    with open(MODELS_DIR / "ieee_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved to {MODELS_DIR}/:")
    print(f"   ieee_fraud_model.joblib")
    print(f"   ieee_scaler.joblib")
    print(f"   ieee_feature_names.json")
    print(f"   ieee_model_metadata.json")


def main():
    df = load_data()
    df = engineer_features(df)
    X, y, feature_names = preprocess(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Scaler (for any normalized features if needed — stored for consistency)
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = train(X_train, y_train, X_val, y_val)
    auc, prauc, threshold, top_features = evaluate(model, X_test, y_test, feature_names)
    save_artifacts(model, scaler, feature_names, auc, prauc, threshold, top_features)


if __name__ == "__main__":
    main()
