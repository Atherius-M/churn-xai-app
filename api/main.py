from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import explain as explain_mod
from src import export as export_mod

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if export_mod.model_exists():
        _load_model()
    yield


app = FastAPI(
    title="Churn XAI - API d'interpretation",
    description=(
        "Service FastAPI expose les predictions et les explications (SHAP, LIME) "
        "du meilleur modele de classification du churn Telco, pour l'application Streamlit."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _load_model():
    pipeline, metadata, X_background, X_sample, y_sample = export_mod.load_exported_model()
    STATE["pipeline"] = pipeline
    STATE["metadata"] = metadata
    STATE["X_background"] = X_background
    STATE["X_sample"] = X_sample
    STATE["y_sample"] = y_sample
    STATE["shap_explainer"] = explain_mod.build_shap_explainer(pipeline, X_background)
    STATE["lime_explainer"] = explain_mod.build_lime_explainer(
        X_background, metadata["numeric_cols"], metadata["categorical_cols"],
        class_names=["No", "Yes"],
    )
    STATE["feature_names_transformed"] = explain_mod.get_transformed_feature_names(pipeline)


def _require_model():
    if "pipeline" not in STATE:
        raise HTTPException(status_code=503, detail="Aucun modele exporte. Entrainez et exportez un modele depuis l'application Streamlit (Module 4).")


class PredictRequest(BaseModel):
    features: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in STATE}


@app.get("/schema")
def schema():
    _require_model()
    metadata = STATE["metadata"]
    example = STATE["X_background"].iloc[0].to_dict()
    return {**metadata, "example_row": example}


@app.post("/predict")
def predict(req: PredictRequest):
    _require_model()
    metadata = STATE["metadata"]
    row = pd.DataFrame([req.features])[metadata["feature_cols"]]
    proba = float(STATE["pipeline"].predict_proba(row)[0, 1])
    decision = int(proba >= metadata["threshold"])
    return {"probability": proba, "decision": decision, "threshold": metadata["threshold"]}


@app.post("/explain/local")
def explain_local(req: PredictRequest):
    _require_model()
    metadata = STATE["metadata"]
    feature_cols = metadata["feature_cols"]
    row_df = pd.DataFrame([req.features])[feature_cols]
    row = row_df.iloc[0]

    proba = float(STATE["pipeline"].predict_proba(row_df)[0, 1])
    decision = int(proba >= metadata["threshold"])

    shap_out = explain_mod.shap_values_for(STATE["shap_explainer"], STATE["pipeline"], row_df)
    shap_row = pd.Series(shap_out.values[0], index=STATE["feature_names_transformed"])

    lime_dict = explain_mod.lime_explanation_for(
        STATE["lime_explainer"], STATE["pipeline"], row
    )

    synthesis = explain_mod.synthesis_sentence(shap_row, proba, metadata["threshold"])

    return {
        "probability": proba,
        "decision": decision,
        "threshold": metadata["threshold"],
        "shap": shap_row.to_dict(),
        "lime": lime_dict,
        "synthesis": synthesis,
    }


@app.get("/explain/global")
def explain_global():
    _require_model()
    pipeline = STATE["pipeline"]
    metadata = STATE["metadata"]
    X_sample = STATE["X_sample"]
    y_sample = STATE["y_sample"]

    native = explain_mod.native_importance(pipeline, STATE["feature_names_transformed"])
    perm = explain_mod.permutation_importances(pipeline, X_sample, y_sample, n_repeats=3)

    shap_sample = X_sample.sample(min(150, len(X_sample)), random_state=42)
    shap_out = explain_mod.shap_values_for(STATE["shap_explainer"], pipeline, shap_sample)
    transformed_names = STATE["feature_names_transformed"]
    transformed_values = pd.DataFrame(
        explain_mod.transform_dense(pipeline, shap_sample), columns=transformed_names
    )

    return {
        "native_importance": native.to_dict(),
        "permutation_importance": perm.to_dict(),
        "shap_feature_names": transformed_names,
        "shap_values": shap_out.values.tolist(),
        "shap_feature_values": transformed_values.to_dict(orient="list"),
    }
