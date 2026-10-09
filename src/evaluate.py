import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics_row(model_name: str, y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    return {
        "modele": model_name,
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "PR-AUC": pr_auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "rappel": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def metrics_table(models_proba: dict, y_true, threshold: float = 0.5) -> pd.DataFrame:
    rows = [compute_metrics_row(name, y_true, proba, threshold) for name, proba in models_proba.items()]
    return pd.DataFrame(rows).set_index("modele").round(4)


def roc_overlay_figure(models_proba: dict, y_true) -> go.Figure:
    fig = go.Figure()
    for name, proba in models_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_score = roc_auc_score(y_true, proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_score:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Hasard"))
    fig.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", title="Courbes ROC")
    return fig


def pr_curve_figure(models_proba: dict, y_true) -> go.Figure:
    fig = go.Figure()
    for name, proba in models_proba.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    fig.update_layout(xaxis_title="Rappel", yaxis_title="Precision", title="Courbes precision-rappel")
    return fig


def confusion_matrix_figure(y_true, y_proba, threshold: float) -> go.Figure:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non-churn (0)", "Churn (1)"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}", colorscale="Blues",
    ))
    fig.update_layout(title="Matrice de confusion", xaxis_title="Predit", yaxis_title="Reel")
    return fig


def threshold_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "rappel": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
