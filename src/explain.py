import re

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

    # LIME ne gere pas les NaN (ex. TotalCharges vide pour tenure = 0) : ses statistiques de
    # discretisation deviendraient NaN et explain_instance planterait. On impute par la mediane,
    # comme le fait deja le pipeline du modele.
    numeric_medians = {
        c: float(pd.to_numeric(X_background[c], errors="coerce").median()) for c in numeric_cols
    }

    encoded = X_background.copy()
    for c, cats in category_maps.items():
        encoded[c] = encoded[c].astype(str).map({cat: i for i, cat in enumerate(cats)})
    for c, med in numeric_medians.items():
        encoded[c] = pd.to_numeric(encoded[c], errors="coerce").fillna(med)
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
    return {
        "explainer": explainer,
        "category_maps": category_maps,
        "feature_cols": feature_cols,
        "numeric_medians": numeric_medians,
    }


def _encode_row_for_lime(row: pd.Series, feature_cols: list, category_maps: dict, numeric_medians: dict = None) -> np.ndarray:
    numeric_medians = numeric_medians or {}
    values = []
    for c in feature_cols:
        if c in category_maps:
            cats = category_maps[c]
            val = str(row[c])
            values.append(float(cats.index(val)) if val in cats else 0.0)
        else:
            val = pd.to_numeric(row[c], errors="coerce")
            if pd.isna(val):
                val = numeric_medians.get(c, 0.0)
            values.append(float(val))
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

    encoded_row = _encode_row_for_lime(row, feature_cols, category_maps, lime_bundle.get("numeric_medians"))
    exp = lime_bundle["explainer"].explain_instance(encoded_row, predict_fn, num_features=num_features)
    return dict(exp.as_list())


def _match_original_feature(label: str, feature_cols: list, prefix_only: bool):
    """Retrouve la variable d'origine derriere un nom transforme (SHAP) ou une regle LIME."""
    best = None
    for f in feature_cols:
        if prefix_only:
            hit = label == f or label.startswith(f + "_")
        else:
            hit = re.search(rf"(^|[\s<>=]){re.escape(f)}($|[\s<>=])", label) is not None
        if hit and (best is None or len(f) > len(best)):
            best = f
    return best


def shap_by_original_feature(shap_row: pd.Series, feature_cols: list) -> pd.Series:
    """Somme les contributions SHAP des colonnes one-hot pour revenir aux variables d'origine."""
    totals = {}
    for name, val in shap_row.items():
        f = _match_original_feature(name, feature_cols, prefix_only=True) or name
        totals[f] = totals.get(f, 0.0) + float(val)
    return pd.Series(totals)


def lime_by_original_feature(lime_dict: dict, feature_cols: list) -> pd.Series:
    totals = {}
    for rule, val in lime_dict.items():
        f = _match_original_feature(rule, feature_cols, prefix_only=False) or rule
        totals[f] = totals.get(f, 0.0) + float(val)
    return pd.Series(totals)


def compare_shap_lime(shap_row: pd.Series, lime_dict: dict, feature_cols: list, top_k: int = 5) -> dict:
    """Compare SHAP et LIME au niveau des variables d'origine : rang, signe, recouvrement du top-k."""
    shap_f = shap_by_original_feature(shap_row, feature_cols)
    lime_f = lime_by_original_feature(lime_dict, feature_cols)
    shap_rank = shap_f.abs().rank(ascending=False, method="first")
    lime_rank = lime_f.abs().rank(ascending=False, method="first")

    rows = []
    for f in shap_f.abs().sort_values(ascending=False).index:
        in_lime = f in lime_f.index
        rows.append({
            "variable": f,
            "shap": float(shap_f[f]),
            "rang_shap": int(shap_rank[f]),
            "lime": float(lime_f[f]) if in_lime else None,
            "rang_lime": int(lime_rank[f]) if in_lime else None,
            "meme_signe": bool(np.sign(shap_f[f]) == np.sign(lime_f[f])) if in_lime else None,
        })

    top_shap = set(shap_f.abs().sort_values(ascending=False).head(top_k).index)
    top_lime = set(lime_f.abs().sort_values(ascending=False).head(top_k).index)
    common = [f for f in lime_f.index if f in shap_f.index]
    same_sign = sum(np.sign(shap_f[f]) == np.sign(lime_f[f]) for f in common)
    if len(common) >= 3:
        rho = float(pd.Series(shap_rank[common]).corr(pd.Series(lime_rank[common]), method="spearman"))
    else:
        rho = None

    return {
        "rows": rows,
        "top_k": top_k,
        "top_k_overlap": len(top_shap & top_lime),
        "top_k_common": sorted(top_shap & top_lime),
        "sign_agreement": int(same_sign),
        "n_common": len(common),
        "rank_spearman": rho,
    }


def comparison_sentence(comp: dict) -> str:
    k, overlap = comp["top_k"], comp["top_k_overlap"]
    n, agree = comp["n_common"], comp["sign_agreement"]
    if overlap >= k - 1 and n and agree == n:
        verdict = "SHAP et LIME sont coherents"
    elif overlap >= k // 2 + 1:
        verdict = "SHAP et LIME sont globalement coherents"
    else:
        verdict = "SHAP et LIME divergent pour cet individu (explication locale a interpreter avec prudence)"
    rho = comp["rank_spearman"]
    rho_txt = f", correlation des rangs {rho:.2f}" if rho is not None else ""
    return (
        f"{verdict} : {overlap}/{k} variables communes dans le top {k}, "
        f"meme signe pour {agree}/{n} variables expliquees par les deux methodes{rho_txt}."
    )


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
