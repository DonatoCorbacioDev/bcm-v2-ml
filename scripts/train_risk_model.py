"""
Train and evaluate the ML risk scoring model.

Reads data/synthetic_contracts.csv (generate it first with generate_training_data.py).
Trains Logistic Regression, Random Forest and XGBoost, reports precision/recall/F1/ROC-AUC,
and saves the best model (by test macro F1) to model/risk_model.joblib.

Run:
    python scripts/generate_training_data.py
    python scripts/train_risk_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_contracts.csv"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
META_PATH = MODEL_DIR / "risk_model_metadata.json"

FEATURES = [
    "days_until_expiry",
    "status_code",
    "has_end_date",
    "total_financial_amount",
    "num_financial_records",
    "financial_std",
    "financial_zscore",
]
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]


def _build_pipelines() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=42,
                class_weight="balanced",
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=42,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", verbosity=0,
            )),
        ]),
    }


def _evaluate(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")

    # Cross-validation (macro F1) on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"CV macro F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full train, evaluate on test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    print("\nTest set — classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    print(f"ROC-AUC (macro OvR): {roc_auc:.4f}")

    return {
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "test_macro_f1": float(macro_f1),
        "test_roc_auc": float(roc_auc),
        "report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "Run: python scripts/generate_training_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} samples from {DATA_PATH}")
    print(f"Class distribution:\n{df['risk_level'].value_counts().sort_index()}\n")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["risk_level"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    pipelines = _build_pipelines()
    results = {}
    for name, pipeline in pipelines.items():
        results[name] = _evaluate(name, pipeline, X_train, X_test, y_train, y_test)
        results[name]["pipeline"] = pipeline

    # Pick best by test macro F1
    best_name = max(
        (k for k in results if k != "pipeline"),
        key=lambda k: results[k]["test_macro_f1"],
    )
    best = results[best_name]

    print(f"\n{'=' * 50}")
    print(f"  Best model: {best_name}")
    print(f"  Test macro F1 : {best['test_macro_f1']:.4f}")
    print(f"  Test ROC-AUC  : {best['test_roc_auc']:.4f}")
    print(f"{'=' * 50}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cv_macro_f1_mean": best["cv_macro_f1_mean"],
        "cv_macro_f1_std": best["cv_macro_f1_std"],
        "test_macro_f1": best["test_macro_f1"],
        "test_roc_auc": best["test_roc_auc"],
        "per_class": {
            cls: {
                "precision": best["report"][cls]["precision"],
                "recall": best["report"][cls]["recall"],
                "f1": best["report"][cls]["f1-score"],
            }
            for cls in CLASS_NAMES
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved -> {META_PATH}")


if __name__ == "__main__":
    main()
