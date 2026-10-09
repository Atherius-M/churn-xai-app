import os
import pandas as pd
import streamlit as st

KAGGLE_HANDLE = "blastchar/telco-customer-churn"
DEFAULT_TARGET = "Churn"
ID_COLUMNS = ["customerID"]


@st.cache_data(show_spinner=False)
def download_telco_dataset() -> pd.DataFrame:
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_HANDLE)
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouve dans {path}")
    return pd.read_csv(os.path.join(path, csv_files[0]))


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if DEFAULT_TARGET in df.columns and set(df[DEFAULT_TARGET].dropna().unique()) <= {"Yes", "No"}:
        df[DEFAULT_TARGET] = df[DEFAULT_TARGET].map({"Yes": 1, "No": 0})
    return df


def load_from_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def class_balance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})
