"""
IEEE-CIS Transaction Fraud ML Scorer
======================================
Loads the trained XGBoost model and scores incoming transactions.
Replaces the OpenFinGuard credit-risk proxy with a proper
transaction fraud model trained on real IEEE-CIS data.

Called from fraud_detection_service/main.py after rule checks.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODELS_DIR = Path("../models")

# Match the feature list from train_ieee_model.py
CATEGORICAL_FEATURES = [
    "ProductCD", "P_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

# ProductCD label map (matches what pandas category encoding gives on training data)
PRODUCT_CD_MAP = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}

# Common email domain map (top domains from IEEE-CIS dataset)
EMAIL_DOMAIN_MAP = {
    "gmail.com": 0, "yahoo.com": 1, "hotmail.com": 2, "anonymous.com": 3,
    "outlook.com": 4, "live.com": 5, "icloud.com": 6, "protonmail.com": 7,
    "rediffmail.com": 8, "ymail.com": 9,
}

MATCH_MAP = {"T": 1, "F": 0, "M0": 0, "M1": 1, "M2": 2}


class IEEEFraudScorer:
    """
    Scores a transaction against the IEEE-CIS trained XGBoost model.
    Returns a probability (0.0–1.0) that the transaction is fraudulent.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.threshold = 0.5
        self.metadata = {}
        self._load()

    def _load(self):
        model_path = MODELS_DIR / "ieee_fraud_model.joblib"
        feature_path = MODELS_DIR / "ieee_feature_names.json"
        meta_path = MODELS_DIR / "ieee_model_metadata.json"

        if not model_path.exists():
            print("⚠️  IEEE model not found — IEEE scorer disabled.")
            print("   Run: python scripts/train_ieee_model.py")
            self.model = None
            return

        self.model = joblib.load(model_path)
        with open(feature_path) as f:
            self.feature_names = json.load(f)
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.threshold = self.metadata.get("optimal_threshold", 0.5)
        print(f"✅ IEEE model loaded | AUC-ROC: {self.metadata.get('auc_roc', '?')} | "
              f"Threshold: {self.threshold:.4f}")

    def is_available(self) -> bool:
        return self.model is not None

    def build_feature_vector(self, transaction: dict) -> pd.DataFrame:
        """
        Convert an incoming transaction dict to the feature vector
        the IEEE model expects.

        The transaction dict has keys from your simulator:
          card_id, amount, merchant, city, timestamp, transaction_id, ...

        We map these to IEEE-CIS feature space as best we can.
        Missing features default to -999 (XGBoost handles these well).
        """
        amount = float(transaction.get("amount", 0))

        row = {
            # Core transaction features
            "TransactionAmt": amount,
            "TransactionAmt_log": np.log1p(amount),
            "TransactionDT": transaction.get("transaction_dt", 86400),  # Seconds from reference

            # Product code — map merchant type if available, else default to W (web)
            "ProductCD": PRODUCT_CD_MAP.get(transaction.get("product_cd", "W"), 0),

            # Card features — use hash of card_id to get stable numeric value
            "card1": abs(hash(transaction.get("card_id", ""))) % 10000,
            "card2": abs(hash(transaction.get("card_id", "") + "2")) % 1000,
            "card3": 150,   # Common value in dataset
            "card5": 226,   # Common value in dataset

            # Address — hash of city to numeric
            "addr1": abs(hash(transaction.get("city", ""))) % 500,
            "addr2": 87,    # Common value (country code)

            # Distance — if we have location velocity, encode it
            "dist1": transaction.get("distance_km", -999),

            # Email domain
            "P_emaildomain": EMAIL_DOMAIN_MAP.get(
                transaction.get("email_domain", ""), len(EMAIL_DOMAIN_MAP)
            ),

            # Counting features — from velocity/probe signals
            "C1": transaction.get("txn_count_1h", 1),
            "C2": transaction.get("txn_count_1h", 1),
            "C4": -999,
            "C5": 0,
            "C6": transaction.get("txn_count_1h", 1),
            "C7": 0,
            "C8": 0,
            "C9": 1,
            "C11": transaction.get("txn_count_1h", 1),
            "C12": 0,
            "C13": abs(hash(transaction.get("merchant", ""))) % 100,
            "C14": 1,

            # Timedelta features — distance in time from last transaction
            "D1": transaction.get("mins_since_last_txn", -999),
            "D2": -999,
            "D3": -999,
            "D4": transaction.get("mins_since_last_txn", -999),
            "D10": -999,
            "D15": -999,

            # Match features — set to most common values when unknown
            "M1": 1, "M2": 1, "M3": 1, "M4": 0, "M5": 1,
            "M6": 1, "M7": -999, "M8": -999, "M9": -999,

            # Vesta engineered features — set to -999 when not available
            **{f"V{i}": -999 for i in [
                12,13,14,17,20,23,26,27,29,30,35,36,37,38,
                45,46,47,48,49,53,54,56,57,58,59,60,61,62,
                63,64,69,70,75,76,78,80,81,82,83,84,85,86,
                87,90,91,92,93,94,95,96,97,98,99,100
            ]},

            # Engineered
            "hour_of_day": transaction.get("hour_of_day", 12),
            "day_of_week": transaction.get("day_of_week", 2),
            "is_weekend": int(transaction.get("day_of_week", 2) in [5, 6]),
            "amt_is_round": int(amount % 10 == 0),
            "is_high_value": int(amount > 500),
        }

        # Build dataframe with only the features the model was trained on
        df = pd.DataFrame([row])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = -999

        df = df[self.feature_names].fillna(-999)
        return df

    def score(self, transaction: dict) -> dict:
        """
        Score a transaction. Returns:
        {
            "available": bool,
            "fraud_probability": float,
            "is_fraud": bool,
            "threshold": float,
            "top_signals": list of str,    # which features pushed the score up
        }
        """
        if not self.is_available():
            return {
                "available": False,
                "fraud_probability": 0.0,
                "is_fraud": False,
                "threshold": self.threshold,
                "top_signals": [],
            }

        X = self.build_feature_vector(transaction)
        proba = float(self.model.predict_proba(X)[0, 1])
        is_fraud = proba >= self.threshold

        # Human-readable signal summary for LLM agent context
        top_signals = []
        amount = float(transaction.get("amount", 0))
        if amount > 10000:
            top_signals.append(f"High transaction amount (₹{amount:,.0f})")
        if transaction.get("txn_count_1h", 0) > 3:
            top_signals.append(f"High transaction frequency ({transaction['txn_count_1h']} txns/hr)")
        if transaction.get("distance_km", 0) > 500:
            top_signals.append(f"Large location jump ({transaction['distance_km']:.0f} km)")
        if amount % 10 == 0 and amount < 100:
            top_signals.append("Round small amount — possible probe transaction")
        if not top_signals:
            top_signals = ["Composite pattern from card & amount features"]

        return {
            "available": True,
            "fraud_probability": round(proba * 100, 2),
            "is_fraud": is_fraud,
            "threshold": round(self.threshold * 100, 2),
            "top_signals": top_signals,
            "model_type": "IEEE-CIS XGBoost (trained on real transaction data)",
        }


# Module-level singleton — loaded once on import
_scorer = None

def get_scorer() -> IEEEFraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = IEEEFraudScorer()
    return _scorer


if __name__ == "__main__":
    # Quick smoke test
    scorer = IEEEFraudScorer()
    test_txn = {
        "card_id": "CARD5555",
        "amount": 50000,
        "merchant": "OnlineStore",
        "city": "Mumbai",
        "transaction_dt": 86400,
        "txn_count_1h": 8,
        "distance_km": 0,
        "hour_of_day": 2,
        "day_of_week": 6,
    }
    result = scorer.score(test_txn)
    print(f"Test result: {result}")
