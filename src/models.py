from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "Regression logistique": {
        "estimator": lambda params: LogisticRegression(max_iter=1000, **params),
        "default_params": {"C": 1.0},
        "search_space": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "Arbre de decision": {
        "estimator": lambda params: DecisionTreeClassifier(random_state=42, **params),
        "default_params": {"max_depth": 5},
        "search_space": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10]},
    },
    "K plus proches voisins": {
        "estimator": lambda params: KNeighborsClassifier(**params),
        "default_params": {"n_neighbors": 15},
        "search_space": {"n_neighbors": [5, 11, 15, 21, 31]},
    },
    "Random Forest": {
        "estimator": lambda params: RandomForestClassifier(random_state=42, **params),
        "default_params": {"n_estimators": 300, "max_depth": None},
        "search_space": {"n_estimators": [200, 300, 500], "max_depth": [5, 10, 20, None]},
    },
    "XGBoost": {
        "estimator": lambda params: XGBClassifier(
            random_state=42, eval_metric="logloss", **params
        ),
        "default_params": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
        "search_space": {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.2],
        },
    },
}


def build_pipeline(model_name: str, preprocessor, params: dict) -> Pipeline:
    estimator = MODEL_REGISTRY[model_name]["estimator"](params)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def train_model(model_name: str, preprocessor, X_train, y_train, params: dict, optimize: bool, n_iter: int = 10):
    pipeline = build_pipeline(model_name, preprocessor, {} if optimize else params)

    if not optimize:
        pipeline.fit(X_train, y_train)
        return pipeline, params

    search_space = MODEL_REGISTRY[model_name]["search_space"]
    param_distributions = {f"model__{k}": v for k, v in search_space.items()}
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter, _space_size(search_space)),
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def _space_size(search_space: dict) -> int:
    size = 1
    for values in search_space.values():
        size *= len(values)
    return size
