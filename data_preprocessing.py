from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split


def load_data(sav_path: Path) -> pd.DataFrame:
    cols = [
        "V005", "V024", "V025", "V113", "V116", "V469E",
        "V106", "V190",
        "V414A", "V414E", "V414F", "V414G", "V414I",
        "V414J", "V414K", "V414L", "V414M", "V414N",
        "V414O", "V414P", "V414S",
        "H11", "H22", "H43",
        "HW1", "HW3", "HW5", "HW6", "HW7",
        "HW8", "HW56", "HW57", "HW70", "HW71", "HW72",
        "B4", "SDIST",
    ]
    df, _ = pyreadstat.read_sav(
        str(sav_path),
        usecols=cols,
        apply_value_formats=False,
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["B4"] = pd.to_numeric(out["B4"], errors="coerce")
    out["B4"] = out["B4"].where(out["B4"].isin([1, 2]), other=np.nan)
    out["B4"] = out["B4"].fillna(out["B4"].dropna().mode().iloc[0])
    out["sex"] = out["B4"].map({1: 0, 2: 1})

    for col in ["HW5", "HW70", "HW8"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([9998, 9980, 9996, 9997, 9999], np.nan) / 100.0
        out.loc[out[col].abs() > 6, col] = np.nan
    out.rename(columns={"HW8": "HAZ_alt"}, inplace=True)

    out["HW3"] = pd.to_numeric(out["HW3"], errors="coerce").replace([9996, 9997, 9998, 9999], np.nan)
    if out["HW3"].dropna().median() > 100:
        out["HW3"] = out["HW3"] / 100.0
    out.loc[(out["HW3"] < 1.5) | (out["HW3"] > 30), "HW3"] = np.nan

    out["HW56"] = pd.to_numeric(out["HW56"], errors="coerce").replace([999, 9998, 9999], np.nan)
    if out["HW56"].dropna().median() > 50:
        out["HW56"] = out["HW56"] / 10.0
    out.loc[(out["HW56"] < 3) | (out["HW56"] > 20), "HW56"] = np.nan

    out["HW57"] = pd.to_numeric(out["HW57"], errors="coerce").replace([8, 9], np.nan)
    out["HW57"] = out["HW57"].map({1: 0, 2: 1, 3: 2, 4: 3})

    for col in ["HW6", "HW7", "HW71", "HW72"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].where(out[col].isin([0, 1]), other=np.nan)

    if out["HW6"].isna().all() or out["HW6"].sum() == 0:
        out["HW6"] = (out["HW5"] < -2).astype(float)
    if out["HW7"].isna().all() or out["HW7"].sum() == 0:
        out["HW7"] = (out["HW5"] < -3).astype(float)
    if out["HW71"].isna().all() or out["HW71"].sum() == 0:
        out["HW71"] = (out["HW70"] < -2).astype(float)
    if out["HW72"].isna().all() or out["HW72"].sum() == 0:
        out["HW72"] = (out["HW5"] < -2).astype(float)

    out["H11"] = pd.to_numeric(out["H11"], errors="coerce").replace([8, 9], np.nan).map({0: 0, 2: 1})
    for col in ["H22", "H43"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([8, 9], np.nan)

    food_cols = [c for c in out.columns if c.startswith("V414")]
    for col in food_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([8, 9], np.nan)

    out["HW1"] = pd.to_numeric(out["HW1"], errors="coerce")
    for col in food_cols:
        out.loc[out["HW1"] < 6, col] = out.loc[out["HW1"] < 6, col].fillna(0)
        out[col] = out[col].fillna(0)
    out.loc[out["HW1"] < 12, "H43"] = out.loc[out["HW1"] < 12, "H43"].fillna(0)
    out["H43"] = out["H43"].fillna(0)

    out["age_under_6"] = (out["HW1"] < 6).astype(int)
    out["age_under_12"] = (out["HW1"] < 12).astype(int)
    out["age_under_24"] = (out["HW1"] < 24).astype(int)

    for col in ["HW1", "HW3", "HW5", "HW70", "HAZ_alt", "HW56"]:
        out[col] = out[col].fillna(out[col].median())

    for col in ["sex", "HW6", "HW7", "HW57", "HW71", "HW72", "V025", "V106", "V190", "V024", "H11", "H22", "V116", "V469E"]:
        valid = out[col].dropna()
        out[col] = out[col].fillna(0 if len(valid) == 0 else valid.mode().iloc[0])

    return out


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["malnutrition_type"] = np.select(
        [
            out["HW5"] < -3,
            (out["HW5"] >= -3) & (out["HW5"] < -2),
            (out["HW5"] >= -2) & (out["HW70"] < -2),
            (out["HW5"] >= -2) & (out["HW70"] >= -2),
        ],
        ["SAM", "MAM", "Stunted", "Normal"],
        default="Unknown",
    )

    out["iron_risk"] = ((out["HW57"] >= 2) | (out["HW56"] < 10.0)).astype(int)
    out["vita_risk"] = ((out["H22"] == 0) & (out["HW70"] < -2)).astype(int)
    out["pem_risk"] = (out["HW5"] < -3).astype(int)

    out["deficiency_type"] = np.select(
        [out["pem_risk"] == 1, out["iron_risk"] == 1, out["vita_risk"] == 1],
        ["Protein_Energy", "Iron_Anaemia", "Vitamin_A"],
        default="None",
    )

    whz_component = (-out["HW5"]).clip(lower=0, upper=4) * 10
    haz_component = (-out["HW70"]).clip(lower=0, upper=(30 / 7)) * 7
    anaemia_component = out["HW57"].map({0: 0, 1: 10, 2: 20, 3: 30}).fillna(0)
    out["severity_score"] = (whz_component + haz_component + anaemia_component).clip(0, 100).round(1)

    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["age_months"] = pd.to_numeric(df["HW1"], errors="coerce")
    x["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    x["rural"] = pd.to_numeric(df["V025"], errors="coerce").map({1: 0, 2: 1})
    x["weight_kg"] = pd.to_numeric(df["HW3"], errors="coerce")
    x["height_cm"] = pd.to_numeric(df["HAZ_alt"], errors="coerce")
    x["waz"] = np.nan
    x["whz"] = pd.to_numeric(df["HW5"], errors="coerce")
    x["haz"] = pd.to_numeric(df["HW70"], errors="coerce")
    x["haemoglobin"] = pd.to_numeric(df["HW56"], errors="coerce")
    x["wealth_index"] = pd.to_numeric(df["V190"], errors="coerce")
    x["mother_edu"] = pd.to_numeric(df["V106"], errors="coerce")

    fuel = pd.to_numeric(df["V469E"], errors="coerce")
    x["clean_fuel"] = fuel.isin([1, 2, 3]).astype(int)
    water = pd.to_numeric(df["V113"], errors="coerce")
    x["improved_water"] = water.isin([11, 12, 13, 14, 21, 31, 41, 51, 71, 72]).astype(int)
    sani = pd.to_numeric(df["V116"], errors="coerce")
    x["improved_sanitation"] = sani.isin([11, 12, 13, 14, 15, 21, 22, 41]).astype(int)

    x["fever_2wks"] = pd.to_numeric(df["H11"], errors="coerce").fillna(0)
    x["vit_a_suppl"] = pd.to_numeric(df["H22"], errors="coerce").fillna(0)
    x["deworming"] = pd.to_numeric(df["H43"], errors="coerce").fillna(0)

    food_map = {
        "V414A": "food_grain", "V414B": "food_legume", "V414C": "food_dairy", "V414D": "food_flesh",
        "V414E": "food_egg", "V414F": "food_vita_veg", "V414G": "food_other_veg", "V414H": "food_vita_fruit",
        "V414I": "food_other_fruit", "V414J": "food_organ", "V414K": "food_processed", "V414L": "food_sweet_drink",
        "V414M": "food_nuts", "V414N": "food_breastmilk", "V414O": "food_formula", "V414P": "food_thin_porridge",
        "V414Q": "food_thick_porridge", "V414R": "food_fortified", "V414S": "food_other",
    }
    for src, dst in food_map.items():
        x[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).astype(int) if src in df.columns else 0

    diversity_cols = ["food_grain", "food_legume", "food_dairy", "food_flesh", "food_egg", "food_vita_veg", "food_other_veg", "food_vita_fruit"]
    x["dietary_diversity"] = x[diversity_cols].sum(axis=1)
    x["anaemia_level"] = pd.to_numeric(df["HW57"], errors="coerce").fillna(0)

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    return x


def save_processed(x: pd.DataFrame, y_class: pd.Series, y_score: pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = y_class.notna() & (y_class != "Unknown")
    x_clean = x.loc[mask].copy()
    y_clean = y_class.loc[mask].copy()
    y_score_clean = y_score.loc[mask].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    y_score_train = y_score_clean.loc[x_train.index]
    y_score_val = y_score_clean.loc[x_val.index]
    y_score_test = y_score_clean.loc[x_test.index]

    x_train.to_csv(out_dir / "X_train.csv", index=False)
    x_val.to_csv(out_dir / "X_val.csv", index=False)
    x_test.to_csv(out_dir / "X_test.csv", index=False)

    y_train.rename("deficiency_type").to_csv(out_dir / "y_train.csv", index=False)
    y_val.rename("deficiency_type").to_csv(out_dir / "y_val.csv", index=False)
    y_test.rename("deficiency_type").to_csv(out_dir / "y_test.csv", index=False)

    y_score_train.rename("severity_score").to_csv(out_dir / "y_score_train.csv", index=False)
    y_score_val.rename("severity_score").to_csv(out_dir / "y_score_val.csv", index=False)
    y_score_test.rename("severity_score").to_csv(out_dir / "y_score_test.csv", index=False)


def main():
    root = Path(__file__).resolve().parent
    sav_path = root / "IAKR7EFL.SAV"
    if not sav_path.exists():
        raise FileNotFoundError(f"Dataset not found: {sav_path}")

    df = load_data(sav_path)
    df = clean_data(df)
    df = create_targets(df)
    x = build_feature_matrix(df)

    save_processed(
        x=x,
        y_class=df["deficiency_type"],
        y_score=df["severity_score"],
        out_dir=root / "processed",
    )
    print("Saved processed train/val/test CSV files to processed\\")


if __name__ == "__main__":
    main()

