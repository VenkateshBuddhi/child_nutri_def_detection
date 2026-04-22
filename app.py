from pathlib import Path
from typing import Any, Dict, List
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Malnutrition Prediction API", version="1.0.0")

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

clf = joblib.load(MODELS / "clf_final.pkl")
reg = joblib.load(MODELS / "reg_final.pkl")
imputer = joblib.load(MODELS / "imputer.pkl")
feature_cols = joblib.load(MODELS / "feature_columns.pkl")

with open(MODELS / "model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
classes = metadata.get("classes", [])


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]


def build_model_input(X_raw: pd.DataFrame) -> pd.DataFrame:
    X = X_raw.reindex(columns=feature_cols).copy()
    stats = np.asarray(getattr(imputer, "statistics_", []), dtype=float)

    if stats.shape[0] == len(feature_cols):
        for i, col in enumerate(feature_cols):
            stat = stats[i]
            fill_val = 0.0 if np.isnan(stat) else float(stat)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_val)
        return X

    X_arr = imputer.transform(X)
    if X_arr.shape[1] == len(feature_cols):
        return pd.DataFrame(X_arr, columns=feature_cols, index=X.index)

    dropped_idx = [i for i, v in enumerate(stats) if np.isnan(v)]
    kept_cols = [c for i, c in enumerate(feature_cols) if i not in dropped_idx]
    dropped_cols = [feature_cols[i] for i in dropped_idx]

    X_kept = pd.DataFrame(X_arr, columns=kept_cols, index=X.index)
    for c in dropped_cols:
        X_kept[c] = 0.0

    return X_kept[feature_cols]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")

    X_raw = pd.DataFrame(req.rows)
    X = build_model_input(X_raw)

    cls_raw = clf.predict(X)
    cls_pred = [
        classes[int(x)] if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
        for x in cls_raw
    ]
    score_pred = reg.predict(X)

    output = []
    for i, (c, s) in enumerate(zip(cls_pred, score_pred)):
        output.append({
            "row_index": i,
            "pred_deficiency_type": c,
            "pred_severity_score": round(float(s), 2)
        })

    return {"predictions": output}