from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


def decode_class_predictions(preds, classes):
    """Map numeric class predictions to names if needed."""
    if len(preds) == 0:
        return preds
    first = preds[0]
    if isinstance(first, (int, float)):
        return [classes[int(x)] if int(x) < len(classes) else f"UNKNOWN_{x}" for x in preds]
    return preds


def build_model_input(X_raw, feature_cols, imputer):
    """
    Build model-ready features while preserving all expected columns.
    Handles old imputers that drop all-NaN features during transform.
    """
    X = X_raw.reindex(columns=feature_cols).copy()

    # Robust path: use learned statistics directly (keeps full feature count)
    if hasattr(imputer, "statistics_"):
        stats = np.asarray(imputer.statistics_, dtype=float)
        if stats.shape[0] == len(feature_cols):
            for i, col in enumerate(feature_cols):
                stat = stats[i]
                fill_val = 0.0 if np.isnan(stat) else float(stat)
                X[col] = X[col].fillna(fill_val)
            return X

    # Fallback path: transform then restore dropped all-NaN columns
    X_arr = imputer.transform(X)
    if X_arr.shape[1] == len(feature_cols):
        return pd.DataFrame(X_arr, columns=feature_cols, index=X.index)

    # Old SimpleImputer behavior: all-NaN columns are dropped on transform.
    stats = np.asarray(getattr(imputer, "statistics_", []), dtype=float)
    dropped_idx = [i for i, v in enumerate(stats) if np.isnan(v)]
    kept_cols = [c for i, c in enumerate(feature_cols) if i not in dropped_idx]
    dropped_cols = [feature_cols[i] for i in dropped_idx]

    X_kept = pd.DataFrame(X_arr, columns=kept_cols, index=X.index)
    for c in dropped_cols:
        X_kept[c] = 0.0

    return X_kept[feature_cols]


def main():
    base = Path(__file__).resolve().parent

    # Load artifacts
    clf = joblib.load(base / "models" / "clf_final.pkl")
    reg = joblib.load(base / "models" / "reg_final.pkl")
    imputer = joblib.load(base / "models" / "imputer.pkl")
    feature_cols = joblib.load(base / "models" / "feature_columns.pkl")

    with open(base / "models" / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    classes = metadata.get("classes", [])

    # Load 5 sample rows from test split
    X_test = pd.read_csv(base / "processed" / "X_test.csv")
    y_test_cls = pd.read_csv(base / "processed" / "y_test.csv")["deficiency_type"]
    y_test_score = pd.read_csv(base / "processed" / "y_score_test.csv")["severity_score"]

    sample_idx = [0, 633, 582, 53, 434]  # 5 sample indices from test set
    X_sample = X_test.iloc[sample_idx].copy()
    y_true_cls = y_test_cls.iloc[sample_idx].reset_index(drop=True)
    y_true_score = y_test_score.iloc[sample_idx].reset_index(drop=True)

    # Build model-ready input with exact expected feature count/order
    X_sample_imp = build_model_input(X_sample, feature_cols, imputer)

    # Predictions
    pred_cls_raw = clf.predict(X_sample_imp)
    pred_cls = decode_class_predictions(list(pred_cls_raw), classes)
    pred_score = reg.predict(X_sample_imp)

    # Report
    results = pd.DataFrame(
        {
            "row_id": sample_idx,
            "true_deficiency_type": y_true_cls,
            "pred_deficiency_type": pred_cls,
            "true_severity_score": y_true_score.round(2),
            "pred_severity_score": pd.Series(pred_score).round(2),
        }
    )
    results["abs_error_score"] = (
        (results["true_severity_score"] - results["pred_severity_score"]).abs().round(2)
    )

    print("\n=== 5-SAMPLE PREDICTION CHECK ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
