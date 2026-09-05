"""
Train and evaluate the ML risk scoring model.

Reads data/synthetic_contracts.csv (generate it first with generate_training_data.py).
Trains a small grid of Logistic Regression / Random Forest / XGBoost
hyperparameter variants, each with SMOTE oversampling of the minority
classes (MEDIUM, HIGH) applied only inside training folds. Selects the best
variant by macro-F1 on a held-out VALIDATION split (never the test split),
calibrates its probabilities, and reports final metrics against two
baselines (majority-class and the existing rule-based score) on the TEST
split - touched exactly once, after model selection is already final.

Run:
    python scripts/generate_training_data.py
    python scripts/train_risk_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

# app/ is a sibling package, not installed — make it importable so FEATURES/
# CLASS_NAMES can't drift from what ml_risk_scoring.py uses at serving time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.risk_features import FEATURES, CLASS_NAMES  # noqa: E402

DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_contracts.csv"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
META_PATH = MODEL_DIR / "risk_model_metadata.json"


def _build_candidates() -> dict:
    """A small hyperparameter grid per model family (9 candidates total),
    each wrapped as an imblearn Pipeline with a SMOTE step first. imblearn's
    Pipeline (unlike sklearn's) only applies SMOTE during .fit(), never at
    .predict()/.predict_proba() time - and since it's inside the pipeline,
    cross_val_score and CalibratedClassifierCV each refit SMOTE independently
    per fold, using only that fold's own training rows. This oversamples the
    minority classes (MEDIUM, HIGH) without ever leaking a synthetic
    neighbor's information into a held-out fold, validation, or test."""
    candidates = {}

    for C in (0.1, 1.0, 10.0):
        candidates[f"LogisticRegression(C={C})"] = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=C, class_weight="balanced")),
        ])

    for n_estimators, max_depth in ((200, 8), (200, 12), (400, None)):
        candidates[f"RandomForest(n={n_estimators},depth={max_depth})"] = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42,
                class_weight="balanced", n_jobs=-1,
            )),
        ])

    for max_depth, learning_rate in ((4, 0.1), (6, 0.1), (6, 0.05)):
        candidates[f"XGBoost(depth={max_depth},lr={learning_rate})"] = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                random_state=42, eval_metric="mlogloss", verbosity=0,
            )),
        ])

    return candidates


def _group_train_val_test_split(X, y, groups, val_size=0.20, test_size=0.20, seed=42):
    """Splits by `groups` (organization_id) so no organization's contracts
    appear in more than one split - a random per-row split would leak
    information between splits via the shared org-level z-score baseline."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(X[train_val_idx], y[train_val_idx], groups[train_val_idx])
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]
    return train_idx, val_idx, test_idx


def _rule_based_baseline_predict(X: np.ndarray) -> np.ndarray:
    """Reimplements app/services/risk_scoring.py's deterministic rule
    (0.6*expiry_score + 0.4*min(|z|/3, 1) -> HIGH/MEDIUM/LOW) directly on
    feature rows, as a baseline the trained model must actually beat. Not
    imported from risk_scoring.py, which pulls in the SQLAlchemy models/DB
    config this standalone script must not depend on - if the production
    rule's weights/thresholds ever change, update both by hand."""
    days = X[:, FEATURES.index("days_until_expiry")]
    has_end_date = X[:, FEATURES.index("has_end_date")]
    z = X[:, FEATURES.index("financial_zscore")]

    expiry_score = np.select(
        [has_end_date == 0, days < 0, days < 30, days < 90, days < 180],
        [0.3, 1.0, 0.8, 0.5, 0.3],
        default=0.1,
    )
    value_score = np.minimum(np.abs(z) / 3.0, 1.0)
    risk_score = 0.6 * expiry_score + 0.4 * value_score

    labels = np.zeros(len(X), dtype=int)
    labels[risk_score >= 0.35] = 1
    labels[risk_score >= 0.65] = 2
    return labels


def _report_baseline(name: str, y_pred: np.ndarray, y_test: np.ndarray) -> dict:
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n{name} baseline — macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return {"macro_f1": float(macro_f1)}


def _select_best_on_validation(candidates: dict, X_train, y_train, X_val, y_val) -> tuple[str, dict]:
    print(f"\n{'=' * 60}\n  MODEL SELECTION (on validation split, test untouched)\n{'=' * 60}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        pipeline.fit(X_train, y_train)
        val_f1 = f1_score(y_val, pipeline.predict(X_val), average="macro")
        scores[name] = {
            "cv_macro_f1_mean": float(cv_scores.mean()),
            "cv_macro_f1_std": float(cv_scores.std()),
            "val_macro_f1": float(val_f1),
        }
        print(f"{name:<32} CV macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}   "
              f"Validation macro F1: {val_f1:.4f}")

    best_name = max(scores, key=lambda k: scores[k]["val_macro_f1"])
    print(f"\nSelected: {best_name} (highest validation macro F1)")
    return best_name, scores


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
    groups = df["organization_id"].to_numpy()

    train_idx, val_idx, test_idx = _group_train_val_test_split(X, y, groups)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(
        f"Train: {len(X_train):,} ({df.iloc[train_idx]['organization_id'].nunique()} orgs)  |  "
        f"Validation: {len(X_val):,} ({df.iloc[val_idx]['organization_id'].nunique()} orgs)  |  "
        f"Test: {len(X_test):,} ({df.iloc[test_idx]['organization_id'].nunique()} orgs)"
    )

    # --- Baselines, evaluated on test exactly like the final model ---
    print(f"\n{'=' * 60}\n  BASELINES\n{'=' * 60}")
    majority = DummyClassifier(strategy="most_frequent", random_state=42)
    majority.fit(X_train, y_train)
    baselines = {
        "majority_class": _report_baseline("Majority-class", majority.predict(X_test), y_test),
        "rule_based": _report_baseline("Rule-based (risk_scoring.py)", _rule_based_baseline_predict(X_test), y_test),
    }

    # --- Model selection on validation, test never touched until the end ---
    candidates = _build_candidates()
    best_name, selection_scores = _select_best_on_validation(candidates, X_train, y_train, X_val, y_val)

    # Refit the winner on train+validation (more data for the final model)
    # then calibrate its probabilities via internal cross-validation on that
    # same combined set - calibration never sees the test split either.
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_pipeline = _build_candidates()[best_name]
    calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=5)
    calibrated.fit(X_train_val, y_train_val)

    # --- Final evaluation, on test, exactly once ---
    print(f"\n{'=' * 60}\n  FINAL MODEL: {best_name} (calibrated) — TEST SET\n{'=' * 60}")
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    y_test_binarized = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
    brier = float(np.mean(np.sum((y_prob - y_test_binarized) ** 2, axis=1)))
    logloss = float(log_loss(y_test, y_prob, labels=list(range(len(CLASS_NAMES)))))
    print(f"Test macro F1  : {macro_f1:.4f}")
    print(f"Test ROC-AUC   : {roc_auc:.4f}")
    print(f"Test Brier     : {brier:.4f}  (lower is better, 0 = perfect calibration)")
    print(f"Test log-loss  : {logloss:.4f}")
    print(f"\nBaselines for comparison — majority-class F1: {baselines['majority_class']['macro_f1']:.4f}, "
          f"rule-based F1: {baselines['rule_based']['macro_f1']:.4f}")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    metadata = {
        "model_name": f"{best_name} (isotonic-calibrated)",
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "split": "grouped by organization_id (no organization spans more than one split)",
        "model_selection": selection_scores,
        "baselines": baselines,
        "test_macro_f1": float(macro_f1),
        "test_roc_auc": float(roc_auc),
        "test_brier_score": brier,
        "test_log_loss": logloss,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
            }
            for cls in CLASS_NAMES
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved -> {META_PATH}")


if __name__ == "__main__":
    main()
