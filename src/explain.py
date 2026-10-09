import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance


def native_importance(pipeline, feature_names: list) -> pd.Series:
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def _clean_name(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def get_transformed_feature_names(pipeline) -> list:
    raw = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [_clean_name(n) for n in raw]


def permutation_importances(pipeline, X, y, n_repeats: int = 5, random_state: int = 42) -> pd.Series:
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc", n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def transform_dense(pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed = pipeline.named_steps["preprocess"].transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def build_shap_explainer(pipeline, X_background: pd.DataFrame):
    model = pipeline.named_steps["model"]
    background = transform_dense(pipeline, X_background)
    background = shap.utils.sample(background, min(100, len(background)), random_state=42)
    predict_fn = lambda data: model.predict_proba(data)[:, 1]
    return shap.Explainer(predict_fn, background)


def shap_values_for(explainer, pipeline, X: pd.DataFrame):
    return explainer(transform_dense(pipeline, X))


def build_lime_explainer(X_background: pd.DataFrame, numeric_cols: list, categorical_cols: list, class_names: list):
    feature_cols = list(X_background.columns)
    categorical_idx = [feature_cols.index(c) for c in categorical_cols]
    category_maps = {c: sorted(X_background[c].astype(str).unique().tolist()) for c in categorical_cols}
    categorical_names = {feature_cols.index(c): cats for c, cats in category_maps.items()}

    encoded = X_background.copy()
    for c, cats in category_maps.items():
        encoded[c] = encoded[c].astype(str).map({cat: i for i, cat in enumerate(cats)})
    encoded = encoded[feature_cols].astype(float).values

    explainer = LimeTabularExplainer(
        encoded,
        feature_names=feature_cols,
        categorical_features=categorical_idx,
        categorical_names=categorical_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )
    return {"explainer": explainer, "category_maps": category_maps, "feature_cols": feature_cols}


def _encode_row_for_lime(row: pd.Series, feature_cols: list, category_maps: dict) -> np.ndarray:
    values = []
    for c in feature_cols:
        if c in category_maps:
            cats = category_maps[c]
            val = str(row[c])
            values.append(float(cats.index(val)) if val in cats else 0.0)
        else:
            values.append(float(row[c]))
    return np.array(values)


def lime_explanation_for(lime_bundle: dict, pipeline, row: pd.Series, num_features: int = 10):
    feature_cols = lime_bundle["feature_cols"]
    category_maps = lime_bundle["category_maps"]

    def predict_fn(encoded_batch: np.ndarray):
        rows = []
        for encoded_row in encoded_batch:
            decoded = {}
            for i, c in enumerate(feature_cols):
                if c in category_maps:
                    cats = category_maps[c]
                    decoded[c] = cats[int(round(encoded_row[i]))]
                else:
                    decoded[c] = encoded_row[i]
            rows.append(decoded)
        return pipeline.predict_proba(pd.DataFrame(rows, columns=feature_cols))

    encoded_row = _encode_row_for_lime(row, feature_cols, category_maps)
    exp = lime_bundle["explainer"].explain_instance(encoded_row, predict_fn, num_features=num_features)
    return dict(exp.as_list())


def synthesis_sentence(shap_row: pd.Series, proba: float, threshold: float, top_k: int = 3) -> str:
    decision = "a risque de depart eleve" if proba >= threshold else "a risque de depart faible"
    ordered = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
    top = ordered.head(top_k)
    reasons = []
    for feat, val in top.items():
        direction = "augmente" if val > 0 else "diminue"
        reasons.append(f"{feat} ({direction} le risque)")
    reasons_text = ", ".join(reasons)
    return f"Ce client {decision} (probabilite {proba:.1%}), principalement a cause de : {reasons_text}."
