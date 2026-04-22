from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score


LEAKAGE_COLS = ["whz", "haz", "anaemia_level"]


def load_splits(processed_dir: Path):
    x_train = pd.read_csv(processed_dir / "X_train.csv")
    x_val = pd.read_csv(processed_dir / "X_val.csv")
    x_test = pd.read_csv(processed_dir / "X_test.csv")

    y_train = pd.read_csv(processed_dir / "y_train.csv")["deficiency_type"]
    y_val = pd.read_csv(processed_dir / "y_val.csv")["deficiency_type"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["deficiency_type"]

    y_score_train = pd.read_csv(processed_dir / "y_score_train.csv")["severity_score"]
    y_score_val = pd.read_csv(processed_dir / "y_score_val.csv")["severity_score"]
    y_score_test = pd.read_csv(processed_dir / "y_score_test.csv")["severity_score"]

    return x_train, x_val, x_test, y_train, y_val, y_test, y_score_train, y_score_val, y_score_test


def remove_leakage(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame):
    existing = [c for c in LEAKAGE_COLS if c in x_train.columns]
    keep_cols = [c for c in x_train.columns if c not in existing]
    return x_train[keep_cols].copy(), x_val[keep_cols].copy(), x_test[keep_cols].copy(), keep_cols, existing


def fit_imputer(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train_imp = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_val_imp = pd.DataFrame(imputer.transform(x_val), columns=x_val.columns, index=x_val.index)
    x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)
    return imputer, x_train_imp, x_val_imp, x_test_imp


def train_models(x_train: pd.DataFrame, y_train_cls: pd.Series, y_train_reg: pd.Series):
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    reg = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(x_train, y_train_cls)
    reg.fit(x_train, y_train_reg)
    return clf, reg


def evaluate(clf, reg, x_val, y_val_cls, y_val_reg, x_test, y_test_cls, y_test_reg):
    val_pred_cls = clf.predict(x_val)
    test_pred_cls = clf.predict(x_test)
    val_pred_reg = reg.predict(x_val)
    test_pred_reg = reg.predict(x_test)

    metrics = {
        "val_macro_f1": round(float(f1_score(y_val_cls, val_pred_cls, average="macro")), 4),
        "test_macro_f1": round(float(f1_score(y_test_cls, test_pred_cls, average="macro")), 4),
        "val_mae_regression": round(float(mean_absolute_error(y_val_reg, val_pred_reg)), 4),
        "test_mae_regression": round(float(mean_absolute_error(y_test_reg, test_pred_reg)), 4),
        "val_r2_regression": round(float(r2_score(y_val_reg, val_pred_reg)), 4),
        "test_r2_regression": round(float(r2_score(y_test_reg, test_pred_reg)), 4),
    }
    return metrics


def main():
    root = Path(__file__).resolve().parent
    processed_dir = root / "processed"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_score_train,
        y_score_val,
        y_score_test,
    ) = load_splits(processed_dir)

    y_train = y_train.fillna("Unknown").astype(str)
    y_val = y_val.fillna("Unknown").astype(str)
    y_test = y_test.fillna("Unknown").astype(str)

    x_train, x_val, x_test, keep_cols, removed_leakage = remove_leakage(x_train, x_val, x_test)
    imputer, x_train_imp, x_val_imp, x_test_imp = fit_imputer(x_train, x_val, x_test)
    clf, reg = train_models(x_train_imp, y_train, y_score_train)
    metrics = evaluate(clf, reg, x_val_imp, y_val, y_score_val, x_test_imp, y_test, y_score_test)

    joblib.dump(clf, models_dir / "clf_final.pkl")
    joblib.dump(reg, models_dir / "reg_final.pkl")
    joblib.dump(imputer, models_dir / "imputer.pkl")
    joblib.dump(keep_cols, models_dir / "feature_columns.pkl")

    metadata = {
        "model_version": "v2_four_class",
        "classes": sorted(y_train.unique().tolist()),
        "n_features": len(keep_cols),
        "feature_names": keep_cols,
        "train_samples": int(len(x_train_imp)),
        "leakage_features_removed": removed_leakage,
        "approach": "Approach B - Basic measurements included",
        "class_distribution_train": y_train.value_counts().to_dict(),
    }
    metadata.update(metrics)

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Saved model files to: {models_dir}")
    print(f"Features used: {len(keep_cols)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

