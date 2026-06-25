# Child Malnutrition Prediction (India NFHS/DHS)

This project predicts:
- **Deficiency type** (classification): `Iron_Anaemia`, `Protein_Energy`, `Vitamin_A`, `None`
- **Severity score** (regression): numeric score (0–100)

It includes:
- Data preprocessing pipeline (`data_preprocessing.py`)
- Model training pipeline (`model_training.py`)
- FastAPI backend (`app.py`)
- Streamlit frontend (`streamlit_app.py`)

---

## 1) Project Structure

```text
project--/
├─ app.py
├─ streamlit_app.py
├─ predict_new_data.py
├─ data_preprocessing.py
├─ model_training.py
├─ data_preprocessing.ipynb
├─ model_training.ipynb
├─ models/                 # generated model artifacts
├─ processed/              # generated train/val/test splits
├─ IAKR7EFL.SAV            # local dataset (not tracked in git)
└─ README.md
```

---

## 2) Setup

### Create and activate environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install pandas numpy pyreadstat scikit-learn fastapi uvicorn[standard] streamlit joblib
```

---

## 3) Data Preprocessing

Make sure dataset file exists in project root:
- `IAKR7EFL.SAV`

Run:

```powershell
python data_preprocessing.py
```

This creates:
- `processed/X_train.csv`, `X_val.csv`, `X_test.csv`
- `processed/y_train.csv`, `y_val.csv`, `y_test.csv`
- `processed/y_score_train.csv`, `y_score_val.csv`, `y_score_test.csv`

---

## 4) Model Training

Run:

```powershell
python model_training.py
```

This creates:
- `models/clf_final.pkl`
- `models/reg_final.pkl`
- `models/imputer.pkl`
- `models/feature_columns.pkl`
- `models/model_metadata.json`

---

## 5) FastAPI Backend

Start server:

```powershell
uvicorn app:app --reload
```

API docs:
- `http://127.0.0.1:8000/docs`

Endpoints:
- `GET /health`
- `POST /predict`

Example request body:

```json
{
  "rows": [
    {
      "age_months": 24,
      "sex": 0,
      "rural": 1,
      "weight_kg": 8.0,
      "haemoglobin": 10.5,
      "wealth_index": 3,
      "mother_edu": 2
    }
  ]
}
```

---

## 6) Streamlit Frontend

Run:

```powershell
streamlit run streamlit_app.py
```

In sidebar, set API URL (default):
- `http://127.0.0.1:8000`

Features:
- Single-row prediction
- Batch CSV upload prediction
- Download predictions as CSV

---

## 7) Quick Inference Script

To test predictions on 5 sample rows:

```powershell
python predict_new_data.py
```

---

## 8) Notes

- Dataset and generated artifacts are excluded using `.gitignore`.
- Keep model and preprocessing versions aligned (retrain after major preprocessing changes).
- For deployment, run FastAPI and Streamlit as separate services.


======================================================================
MODEL TEST RESULTS ON TEST DATA
======================================================================
Classifier - Macro F1 : 0.4464
Regressor  - MAE      : 2.0345
Regressor  - RMSE     : 3.7958
Regressor  - R2       : 0.9661
----------------------------------------------------------------------
Saved metadata values:
test_macro_f1         : 0.7808
test_mae_regression   : 2.0345
test_r2_regression    : 0.9661
----------------------------------------------------------------------
Classification report:
                precision    recall  f1-score   support

  Iron_Anaemia     0.9803    0.9179    0.9481     32428
          None     0.0000    0.0000    0.0000         0
Protein_Energy     0.6718    0.8288    0.7421      4591
     Vitamin_A     0.3898    0.8872    0.5417      1011

     micro avg     0.7399    0.9064    0.8147     38030
     macro avg     0.5105    0.6585    0.5580     38030
  weighted avg     0.9274    0.9064    0.9124     38030

======================================================================