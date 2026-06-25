from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def decode_class_predictions(preds, classes):
    if len(preds) == 0:
        return preds
    first = preds[0]
    if isinstance(first, (int, float, np.integer, np.floating)):
        return [classes[int(x)] if int(x) < len(classes) else f"UNKNOWN_{x}" for x in preds]
    return preds


def build_model_input(x_raw, feature_cols, imputer):
    """
    Build model-ready features while preserving expected columns.
    Handles old imputers that may drop fully-empty columns.
    """
    x = x_raw.reindex(columns=feature_cols).copy()

    if hasattr(imputer, "statistics_"):
        stats = np.asarray(imputer.statistics_, dtype=float)
        if stats.shape[0] == len(feature_cols):
            for i, col in enumerate(feature_cols):
                stat = stats[i]
                fill_val = 0.0 if np.isnan(stat) else float(stat)
                x[col] = pd.to_numeric(x[col], errors="coerce").fillna(fill_val)
            return x

    x_arr = imputer.transform(x)
    if x_arr.shape[1] == len(feature_cols):
        return pd.DataFrame(x_arr, columns=feature_cols, index=x.index)

    stats = np.asarray(getattr(imputer, "statistics_", []), dtype=float)
    dropped_idx = [i for i, v in enumerate(stats) if np.isnan(v)]
    kept_cols = [c for i, c in enumerate(feature_cols) if i not in dropped_idx]
    dropped_cols = [feature_cols[i] for i in dropped_idx]

    x_kept = pd.DataFrame(x_arr, columns=kept_cols, index=x.index)
    for c in dropped_cols:
        x_kept[c] = 0.0

    return x_kept[feature_cols]


def main():
    base = Path(__file__).resolve().parent

    clf = joblib.load(base / "models" / "clf_final.pkl")
    reg = joblib.load(base / "models" / "reg_final.pkl")
    imputer = joblib.load(base / "models" / "imputer.pkl")
    feature_cols = joblib.load(base / "models" / "feature_columns.pkl")

    with open(base / "models" / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    classes = metadata.get("classes", [])

    x_test = pd.read_csv(base / "processed" / "X_test.csv")
    y_test_cls = pd.read_csv(base / "processed" / "y_test.csv")["deficiency_type"].fillna("Unknown").astype(str)
    y_test_score = pd.read_csv(base / "processed" / "y_score_test.csv")["severity_score"]

    x_test_model = build_model_input(x_test, feature_cols, imputer)

    cls_pred_raw = clf.predict(x_test_model)
    cls_pred = pd.Series(decode_class_predictions(list(cls_pred_raw), classes), index=x_test_model.index)
    score_pred = reg.predict(x_test_model)

    macro_f1 = f1_score(y_test_cls, cls_pred, average="macro")
    mae = mean_absolute_error(y_test_score, score_pred)
    rmse = np.sqrt(mean_squared_error(y_test_score, score_pred))
    r2 = r2_score(y_test_score, score_pred)

    print("=" * 70)
    print("MODEL TEST RESULTS ON TEST DATA")
    print("=" * 70)
    print(f"Classifier - Macro F1 : {macro_f1:.4f}")
    print(f"Regressor  - MAE      : {mae:.4f}")
    print(f"Regressor  - RMSE     : {rmse:.4f}")
    print(f"Regressor  - R2       : {r2:.4f}")
    print("-" * 70)
    print("Saved metadata values:")
    print(f"test_macro_f1         : {metadata.get('test_macro_f1')}")
    print(f"test_mae_regression   : {metadata.get('test_mae_regression')}")
    print(f"test_r2_regression    : {metadata.get('test_r2_regression')}")
    print("-" * 70)
    print("Classification report:")
    print(classification_report(y_test_cls, cls_pred, labels=classes, target_names=classes, digits=4, zero_division=0))
    print("=" * 70)


if __name__ == "__main__":
    main()

