import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bar_importance_figure(values: dict, title: str, top_n: int = 15) -> go.Figure:
    series = pd.Series(values).sort_values(ascending=False).head(top_n)
    fig = px.bar(
        x=series.values, y=series.index, orientation="h", title=title,
        labels={"x": "importance", "y": "variable"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def beeswarm_figure(shap_values: list, feature_values: dict, feature_names: list, top_n: int = 12) -> go.Figure:
    shap_arr = np.array(shap_values)
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig = go.Figure()
    rng = np.random.default_rng(42)
    for rank, idx in enumerate(order):
        feat = feature_names[idx]
        vals = shap_arr[:, idx]
        raw = np.array(feature_values[feat])
        if np.issubdtype(raw.dtype, np.number):
            norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            codes = pd.Categorical(raw).codes
            norm = (codes - codes.min()) / (codes.max() - codes.min() + 1e-9)
        jitter = rng.uniform(-0.3, 0.3, size=len(vals))
        fig.add_trace(go.Scatter(
            x=vals, y=[rank + j for j in jitter], mode="markers",
            marker=dict(color=norm, colorscale="RdBu_r", size=6, showscale=(rank == 0),
                        colorbar=dict(title="valeur (norm.)") if rank == 0 else None),
            name=feat, showlegend=False,
            hovertemplate=f"{feat}<br>SHAP=%{{x:.4f}}<extra></extra>",
        ))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                      ticktext=[feature_names[i] for i in order])
    fig.update_layout(title="SHAP summary (beeswarm)", xaxis_title="Impact SHAP sur la prediction")
    return fig


def dependence_figure(shap_values: list, feature_values: dict, feature_names: list, feature: str) -> go.Figure:
    shap_arr = np.array(shap_values)
    idx = feature_names.index(feature)
    x = feature_values[feature]
    y = shap_arr[:, idx]
    fig = px.scatter(x=x, y=y, labels={"x": feature, "y": f"SHAP({feature})"},
                      title=f"Dependance SHAP - {feature}")
    return fig


def waterfall_figure(shap_row: dict, title: str = "SHAP waterfall", top_n: int = 10) -> go.Figure:
    series = pd.Series(shap_row)
    ordered = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
    ordered = ordered.iloc[::-1]
    colors = ["crimson" if v > 0 else "steelblue" for v in ordered.values]
    fig = go.Figure(go.Bar(x=ordered.values, y=ordered.index, orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title="Contribution a la prediction (SHAP)")
    return fig


def lime_bar_figure(lime_dict: dict, title: str = "Explication LIME") -> go.Figure:
    series = pd.Series(lime_dict).sort_values()
    colors = ["crimson" if v > 0 else "steelblue" for v in series.values]
    fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title="Contribution a la prediction (LIME)")
    return fig
