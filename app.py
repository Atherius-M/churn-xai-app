import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src import data as data_mod
from src import evaluate as eval_mod
from src import explain as explain_mod
from src import export as export_mod
from src import models as models_mod
from src import prep as prep_mod
from src import xai_viz as viz_mod

st.set_page_config(page_title="Churn XAI", layout="wide")

API_URL = st.sidebar.text_input("URL de l'API d'interpretation", "http://localhost:8000")

for key in ["df", "target_col", "X_train", "X_test", "y_train", "y_test", "preprocessor", "trained_models"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["trained_models"] is None:
    st.session_state["trained_models"] = {}

page = st.sidebar.radio(
    "Modules",
    ["1. Donnees", "2. Preparation", "3. Modelisation", "4. Evaluation", "5. Explicabilite"],
)

# ---------------------------------------------------------------- Module 1
if page == "1. Donnees":
    st.title("Module 1 - Donnees")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Telecharger automatiquement (Kaggle: Telco Customer Churn)"):
            try:
                df = data_mod.download_telco_dataset()
                st.session_state["df"] = data_mod.clean_telco(df)
                st.success(f"Jeu de donnees charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            except Exception as exc:
                st.error(
                    "Echec du telechargement Kaggle. Verifiez votre authentification "
                    f"kagglehub (~/.kaggle/kaggle.json). Detail : {exc}"
                )
    with col2:
        uploaded = st.file_uploader("...ou importer un CSV", type="csv")
        if uploaded is not None:
            df = data_mod.load_from_csv(uploaded)
            st.session_state["df"] = df
            st.success(f"Fichier importe : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    df = st.session_state["df"]
    if df is not None:
        st.subheader("Variable cible")
        default_idx = list(df.columns).index(data_mod.DEFAULT_TARGET) if data_mod.DEFAULT_TARGET in df.columns else 0
        target_col = st.selectbox("Choisir la variable cible", df.columns, index=default_idx)
        st.session_state["target_col"] = target_col

        st.subheader("Apercu des donnees")
        st.dataframe(df.head(20))

        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all").transpose())

        st.subheader("Valeurs manquantes")
        missing = df.isna().sum()
        st.dataframe(missing[missing > 0].rename("valeurs manquantes"))

        st.subheader("Proportion de chaque classe")
        st.dataframe(data_mod.class_balance(df, target_col))

        st.subheader("Visualisations exploratoires")
        numeric_cols, categorical_cols = prep_mod.split_feature_types(df.drop(columns=[target_col]))

        var = st.selectbox("Variable a explorer", numeric_cols + categorical_cols)
        if var in numeric_cols:
            fig = px.histogram(df, x=var, color=target_col, barmode="overlay", nbins=30)
        else:
            rate = df.groupby(var)[target_col].mean().reset_index()
            fig = px.bar(rate, x=var, y=target_col, title=f"Taux de classe positive par {var}")
        st.plotly_chart(fig, use_container_width=True)

        if numeric_cols:
            st.subheader("Correlations (variables numeriques)")
            corr = df[numeric_cols + [target_col]].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)
    else:
        st.info("Chargez un jeu de donnees pour commencer.")

# ---------------------------------------------------------------- Module 2
elif page == "2. Preparation":
    st.title("Module 2 - Preparation")
    df = st.session_state["df"]
    if df is None:
        st.warning("Retournez au Module 1 pour charger des donnees.")
    else:
        target_col = st.session_state["target_col"] or data_mod.DEFAULT_TARGET
        st.write(f"Variable cible : **{target_col}**")

        candidate_features = [c for c in df.columns if c != target_col]
        excluded = st.multiselect(
            "Colonnes a exclure (identifiants, fuites de donnees...)",
            candidate_features,
            default=[c for c in data_mod.ID_COLUMNS if c in candidate_features],
        )
        feature_cols = [c for c in candidate_features if c not in excluded]

        scale = st.checkbox("Normaliser les variables numeriques (StandardScaler)", value=True)
        test_size = st.slider("Proportion du jeu de test", 0.1, 0.5, 0.2, 0.05)
        stratify = st.checkbox("Stratifier selon la variable cible", value=True)

        if st.button("Appliquer la preparation"):
            work_df = df.dropna(subset=[target_col]).copy()
            X = work_df[feature_cols]
            y_raw = work_df[target_col]
            if pd.api.types.is_numeric_dtype(y_raw):
                y = y_raw.astype(int)
            else:
                classes = sorted(y_raw.unique(), key=str)
                if len(classes) != 2:
                    st.error(f"La variable cible doit etre binaire (trouve {len(classes)} classes : {classes}).")
                    st.stop()
                y = y_raw.map({classes[0]: 0, classes[1]: 1}).astype(int)

            numeric_cols, categorical_cols = prep_mod.split_feature_types(X)
            preprocessor = prep_mod.build_preprocessor(numeric_cols, categorical_cols, scale=scale)

            X_train, X_test, y_train, y_test = prep_mod.make_split(X, y, test_size, stratify)

            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["preprocessor"] = preprocessor
            st.session_state["trained_models"] = {}
            st.session_state["feature_cols"] = feature_cols
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["target_col"] = target_col

            st.success(
                f"Pret : {len(X_train)} lignes d'entrainement, {len(X_test)} lignes de test. "
                f"{len(numeric_cols)} variables numeriques, {len(categorical_cols)} categorielles."
            )

        if st.session_state["X_train"] is not None:
            st.info("Donnees preparees et prêtes pour le Module 3.")

# ---------------------------------------------------------------- Module 3
elif page == "3. Modelisation":
    st.title("Module 3 - Modelisation et comparaison")
    if st.session_state["X_train"] is None:
        st.warning("Completez le Module 2 avant d'entrainer des modeles.")
    else:
        selected_models = st.multiselect(
            "Modeles a entrainer",
            list(models_mod.MODEL_REGISTRY.keys()),
            default=list(models_mod.MODEL_REGISTRY.keys()),
        )
        optimize = st.checkbox("Optimiser les hyperparametres (RandomizedSearchCV)", value=False)

        params_by_model = {}
        if not optimize:
            with st.expander("Hyperparametres manuels"):
                for name in selected_models:
                    st.markdown(f"**{name}**")
                    defaults = models_mod.MODEL_REGISTRY[name]["default_params"]
                    params = {}
                    for p_name, p_default in defaults.items():
                        if isinstance(p_default, bool) or p_default is None:
                            params[p_name] = p_default
                        elif isinstance(p_default, int):
                            params[p_name] = st.number_input(f"{name} - {p_name}", value=p_default, key=f"{name}_{p_name}")
                        elif isinstance(p_default, float):
                            params[p_name] = st.number_input(
                                f"{name} - {p_name}", value=p_default, format="%.4f", key=f"{name}_{p_name}"
                            )
                        else:
                            params[p_name] = p_default
                    params_by_model[name] = params

        if st.button("Entrainer les modeles selectionnes"):
            preprocessor = st.session_state["preprocessor"]
            X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
            trained = {}
            progress = st.progress(0.0)
            for i, name in enumerate(selected_models):
                params = params_by_model.get(name, models_mod.MODEL_REGISTRY[name]["default_params"])
                pipeline, used_params = models_mod.train_model(
                    name, preprocessor, X_train, y_train, params, optimize
                )
                trained[name] = {"pipeline": pipeline, "params": used_params}
                progress.progress((i + 1) / len(selected_models))
            st.session_state["trained_models"] = trained
            st.success(f"{len(trained)} modele(s) entraine(s).")

        trained_models = st.session_state["trained_models"]
        if trained_models:
            st.subheader("Modeles entraines")
            summary = pd.DataFrame(
                {name: info["params"] for name, info in trained_models.items()}
            ).transpose()
            st.dataframe(summary)
            st.caption("Passez au Module 4 pour comparer les performances.")

# ---------------------------------------------------------------- Module 4
elif page == "4. Evaluation":
    st.title("Module 4 - Evaluation")
    trained_models = st.session_state["trained_models"]
    if not trained_models:
        st.warning("Entrainez au moins un modele dans le Module 3.")
    else:
        X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]
        models_proba = {
            name: info["pipeline"].predict_proba(X_test)[:, 1]
            for name, info in trained_models.items()
        }

        threshold = st.slider("Seuil de decision", 0.0, 1.0, 0.5, 0.01)

        st.subheader("Tableau comparatif")
        st.dataframe(eval_mod.metrics_table(models_proba, y_test, threshold))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(eval_mod.roc_overlay_figure(models_proba, y_test), use_container_width=True)
        with col2:
            st.plotly_chart(eval_mod.pr_curve_figure(models_proba, y_test), use_container_width=True)

        st.subheader("Matrice de confusion par modele")
        model_for_cm = st.selectbox("Modele", list(trained_models.keys()))
        st.plotly_chart(
            eval_mod.confusion_matrix_figure(y_test, models_proba[model_for_cm], threshold),
            use_container_width=True,
        )

        st.subheader("Effet du seuil en direct")
        live = eval_mod.threshold_metrics(y_test, models_proba[model_for_cm], threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{live['precision']:.3f}")
        m2.metric("Rappel", f"{live['rappel']:.3f}")
        m3.metric("F1", f"{live['F1']:.3f}")

        st.subheader("Exporter pour le Module 5 / l'API")
        export_choice = st.selectbox("Modele a exporter", list(trained_models.keys()), key="export_choice")
        if st.button("Exporter ce modele"):
            X_train = st.session_state["X_train"]
            background = X_train.sample(min(200, len(X_train)), random_state=42)
            sample = X_test.sample(min(300, len(X_test)), random_state=42)
            export_mod.export_model(
                model_name=export_choice,
                pipeline=trained_models[export_choice]["pipeline"],
                feature_cols=st.session_state["feature_cols"],
                numeric_cols=st.session_state["numeric_cols"],
                categorical_cols=st.session_state["categorical_cols"],
                X_background=background,
                X_sample=sample,
                y_sample=y_test.loc[sample.index],
                target_col=st.session_state["target_col"],
                threshold=threshold,
            )
            st.success(
                f"Modele '{export_choice}' exporte dans models/. "
                "Lancez l'API (uvicorn api.main:app --port 8000) puis ouvrez le Module 5."
            )

# ---------------------------------------------------------------- Module 5
elif page == "5. Explicabilite":
    st.title("Module 5 - Explainable AI")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        api_ready = health.get("model_loaded", False)
    except Exception:
        api_ready = False

    if not api_ready:
        st.warning(
            "API d'interpretation injoignable ou aucun modele exporte. "
            "Exportez un modele (Module 4) et lancez : `uvicorn api.main:app --port 8000` "
            f"(URL configuree : {API_URL})."
        )
        st.stop()

    st.success("API d'interpretation connectee.")

    tab_global, tab_local = st.tabs(["Explications globales", "Explications locales"])

    with tab_global:
        if st.button("Calculer les explications globales"):
            with st.spinner("Calcul SHAP / permutation en cours..."):
                g = requests.get(f"{API_URL}/explain/global", timeout=120).json()
            st.session_state["global_explain"] = g

        g = st.session_state.get("global_explain")
        if g:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    viz_mod.bar_importance_figure(g["native_importance"], "Importance native du modele"),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    viz_mod.bar_importance_figure(g["permutation_importance"], "Importance par permutation"),
                    use_container_width=True,
                )

            st.plotly_chart(
                viz_mod.beeswarm_figure(g["shap_values"], g["shap_feature_values"], g["shap_feature_names"]),
                use_container_width=True,
            )

            top_feature = pd.Series(g["permutation_importance"]).sort_values(ascending=False).index[0]
            feature_choice = st.selectbox(
                "Variable pour le graphique de dependance", g["shap_feature_names"],
                index=g["shap_feature_names"].index(top_feature) if top_feature in g["shap_feature_names"] else 0,
            )
            st.plotly_chart(
                viz_mod.dependence_figure(
                    g["shap_values"], g["shap_feature_values"], g["shap_feature_names"], feature_choice
                ),
                use_container_width=True,
            )

    with tab_local:
        schema = requests.get(f"{API_URL}/schema", timeout=10).json()
        feature_cols = schema["feature_cols"]
        example = schema["example_row"]

        X_test = st.session_state.get("X_test")
        use_test_individual = X_test is not None and st.checkbox("Choisir un individu du jeu de test", value=True)

        if use_test_individual:
            idx = st.selectbox("Index (jeu de test)", X_test.index.tolist())
            row_values = X_test.loc[idx].to_dict()
        else:
            row_values = {}
            cols = st.columns(3)
            for i, feat in enumerate(feature_cols):
                default = example[feat]
                with cols[i % 3]:
                    if isinstance(default, (int, float)):
                        row_values[feat] = st.number_input(feat, value=float(default), key=f"local_{feat}")
                    else:
                        row_values[feat] = st.text_input(feat, value=str(default), key=f"local_{feat}")

        if st.button("Expliquer cette prediction"):
            with st.spinner("Calcul SHAP / LIME en cours..."):
                resp = requests.post(
                    f"{API_URL}/explain/local", json={"features": row_values}, timeout=60
                ).json()

            proba = resp["probability"]
            decision_label = "Churn" if resp["decision"] == 1 else "Non-churn"
            m1, m2 = st.columns(2)
            m1.metric("Probabilite de churn", f"{proba:.1%}")
            m2.metric("Decision (seuil)", decision_label)

            st.info(resp["synthesis"])

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(viz_mod.waterfall_figure(resp["shap"]), use_container_width=True)
            with col2:
                st.plotly_chart(viz_mod.lime_bar_figure(resp["lime"]), use_container_width=True)
