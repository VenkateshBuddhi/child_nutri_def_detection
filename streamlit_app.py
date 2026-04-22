from pathlib import Path
import json
from urllib import request, error

import pandas as pd
import streamlit as st


def post_predict(api_url: str, rows: list[dict]):
    payload = json.dumps({"rows": rows}).encode("utf-8")
    req = request.Request(
        url=f"{api_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_health(api_url: str):
    req = request.Request(
        url=f"{api_url.rstrip('/')}/health",
        method="GET",
    )
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_api_rows(df: pd.DataFrame) -> list[dict]:
    clean_df = df.where(pd.notnull(df), None)
    return clean_df.to_dict(orient="records")


st.set_page_config(page_title="Malnutrition Predictor", layout="wide")
st.title("Child Malnutrition Prediction")
st.caption("Streamlit frontend for FastAPI backend")

default_api = "http://127.0.0.1:8000"
api_url = st.sidebar.text_input("FastAPI URL", value=default_api)

if st.sidebar.button("Check API Health"):
    try:
        health = get_health(api_url)
        st.sidebar.success(f"API status: {health.get('status', 'ok')}")
    except Exception as exc:
        st.sidebar.error(f"API not reachable: {exc}")

feature_names = []
meta_path = Path(__file__).resolve().parent / "models" / "model_metadata.json"
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names", [])

if not feature_names:
    feature_names = [
        "age_months", "sex", "rural", "weight_kg", "height_cm", "waz",
        "haemoglobin", "wealth_index", "mother_edu", "clean_fuel",
        "improved_water", "improved_sanitation", "fever_2wks", "vit_a_suppl",
        "deworming", "food_grain", "food_legume", "food_dairy", "food_flesh",
        "food_egg", "food_vita_veg", "food_other_veg", "food_vita_fruit",
        "food_other_fruit", "food_organ", "food_processed", "food_sweet_drink",
        "food_nuts", "food_breastmilk", "food_formula", "food_thin_porridge",
        "food_thick_porridge", "food_fortified", "food_other", "dietary_diversity",
    ]

default_row = {c: None for c in feature_names}
default_row.update(
    {
        "age_months": 24,
        "sex": 0,
        "rural": 1,
        "weight_kg": 8.0,
        "haemoglobin": 10.5,
        "wealth_index": 3,
        "mother_edu": 2,
        "dietary_diversity": 1,
    }
)

tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV"])

with tab1:
    st.subheader("Single-row prediction")
    st.write("Edit values below and click predict.")
    single_df = st.data_editor(
        pd.DataFrame([default_row]),
        num_rows="fixed",
        use_container_width=True,
        key="single_editor",
    )
    if st.button("Predict Single Row", type="primary"):
        try:
            rows = to_api_rows(single_df)
            resp = post_predict(api_url, rows)
            pred_df = pd.DataFrame(resp.get("predictions", []))
            st.success("Prediction complete")
            st.dataframe(pred_df, use_container_width=True)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            st.error(f"API error ({exc.code}): {detail}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with tab2:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV with one row per child. Missing columns are auto-filled.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("Predict Batch", type="primary"):
            try:
                rows = to_api_rows(batch_df)
                resp = post_predict(api_url, rows)
                pred_df = pd.DataFrame(resp.get("predictions", []))
                out_df = pd.concat(
                    [batch_df.reset_index(drop=True), pred_df.drop(columns=["row_index"], errors="ignore")],
                    axis=1,
                )
                st.success(f"Predicted {len(out_df)} rows")
                st.dataframe(out_df.head(20), use_container_width=True)
                st.download_button(
                    "Download Predictions CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_output.csv",
                    mime="text/csv",
                )
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8")
                st.error(f"API error ({exc.code}): {detail}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
