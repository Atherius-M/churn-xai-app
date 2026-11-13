"""Plain-English explanations shown inside the app: what each page does, what every technical term
means, and how to read every chart and table. Can be switched off from the sidebar."""
import streamlit as st

# --------------------------------------------------------------------------- glossary
GLOSSARY = {
    # data
    "churn": "A customer cancelling their subscription and leaving the company.",
    "target variable (Churn)": "The answer we want to predict: Yes / 1 = the customer left, No / 0 = stayed.",
    "dataset": "The table of customers: one row per customer (7,043), one column per piece of information (21).",
    "Telco Customer Churn": "Public dataset from IBM, on Kaggle: customers of a phone / internet company. "
                            "Case n°13 of the teacher's catalogue.",
    "kagglehub": "A tool that downloads a dataset from the Kaggle website automatically, with no manual step.",
    "CSV": "A simple spreadsheet file (values separated by commas).",
    "tenure": "How many months the person has been a customer.",
    "Contract": "Type of subscription: month-to-month, one year or two years.",
    "MonthlyCharges / TotalCharges": "What the customer pays per month, and has paid in total.",
    "descriptive statistics": "Basic summary numbers per column: count, average, spread, min, max, most frequent value…",
    "missing values (NaN)": "Empty cells. Here 11 new customers (tenure 0) have no TotalCharges yet.",
    "class balance / imbalance": "How the two answers are split: about 26.5 % churn vs 73.5 % stay. "
                                 "Unequal groups = imbalance; it affects which scores we trust.",
    "correlation": "How much two numbers move together, from −1 to +1. +1 = rise together, −1 = one rises when "
                   "the other falls, 0 = no link.",
    # preparation
    "feature / variable": "A piece of information about each customer that the model can use (tenure, contract…).",
    "identifier (customerID)": "A unique code per customer. Meaningless for prediction, and it would let the model "
                               "memorise people, so it is excluded.",
    "data leakage": "Accidentally giving the model information it would not have in real life (e.g. test answers). "
                    "Results look great but are fake.",
    "imputation": "Filling empty cells with a sensible value: the median for numbers, the most frequent value for "
                  "categories.",
    "one-hot encoding": "Turning a category into yes/no columns. Contract = 'One year' becomes three columns: "
                        "Month-to-month = 0, One year = 1, Two year = 0.",
    "normalisation (StandardScaler)": "Putting every number on the same scale (average 0, spread 1) so that a big "
                                      "number like TotalCharges does not look more important just because it is big.",
    "train / test split": "Keep most customers (e.g. 80 %) to LEARN and hide the rest (20 %) to TEST the model "
                          "afterwards, like studying on some questions and sitting the exam on others.",
    "stratification": "Make sure both parts keep the same share of churners (26.5 %), so the exam is fair.",
    "pipeline": "A chain of steps always run in the same order: fill gaps → encode → scale → model. Prevents leakage.",
    # models
    "model": "The 'rule-maker' the computer builds from past customers. It gives each customer a churn probability.",
    "training": "The model studying the training customers to find patterns.",
    "logistic regression": "The simplest model: each piece of information gets a + or − weight, they are added up "
                           "and turned into a probability. Easy to read.",
    "decision tree": "A chain of yes/no questions: 'Month-to-month? → yes → new customer? → yes → likely to leave'.",
    "k-NN (k nearest neighbours)": "Look at the k most similar past customers; if most of them left, this one "
                                   "probably will too.",
    "Random Forest": "Hundreds of decision trees that each vote; the majority wins.",
    "XGBoost": "Many small trees built one after another, each fixing the previous ones' mistakes. Powerful but "
               "harder to read.",
    "hyperparameters": "The model's settings / knobs, chosen before learning.",
    "C": "Logistic regression: how freely the model fits the data (big C = fits more closely).",
    "max_depth": "Trees: how many questions deep a tree can go. Deeper = more detailed but can over-learn.",
    "min_samples_leaf": "Trees: smallest group of customers allowed at the end of a branch.",
    "n_neighbors": "k-NN: how many similar customers are looked at.",
    "n_estimators": "Random Forest / XGBoost: number of trees.",
    "learning_rate": "XGBoost: size of each new tree's correction. Small = slower but safer.",
    "RandomizedSearchCV": "Automatically tries random combinations of settings and keeps the one with the best "
                          "AUC-ROC, using cross-validation.",
    "cross-validation": "Testing several times on different slices of the training data so the choice is not luck.",
    "overfitting": "The model memorised the training customers instead of learning general rules, so it fails on new ones.",
    # evaluation
    "probability": "The model's answer for one customer, from 0 (surely stays) to 1 (surely leaves).",
    "threshold": "The cut-off: above it we predict 'churn'. 0.5 by default; you can move it.",
    "accuracy": "Share of correct answers. Misleading here: always saying 'stays' is already 73.5 % accurate.",
    "precision": "Of the customers we predicted would leave, the share that really left. "
                 "'When I raise the alarm, am I right?'",
    "recall": "Of the customers who really left, the share we caught. 'Did I miss anyone?'",
    "F1": "One score balancing precision and recall (0 = useless, 1 = perfect).",
    "AUC-ROC": "Chance that a random churner gets a higher score than a random loyal customer. 0.5 = coin flip, "
               "1 = perfect. Measures how well the model RANKS customers.",
    "ROC curve": "Graph of caught churners (true positive rate) vs wrongly flagged loyal customers (false positive "
                 "rate) for every threshold.",
    "PR-AUC": "Area under the precision-recall curve. More informative than AUC-ROC when the classes are "
              "unequal, because it focuses on the churners.",
    "confusion matrix": "A 2×2 table of right and wrong answers: correctly caught, false alarms, missed churners, "
                        "correctly left alone.",
    "true / false positive / negative": "True positive = predicted churn and did churn. False positive = predicted "
                                        "churn but stayed. False negative = missed a churner. True negative = "
                                        "correctly predicted 'stays'.",
    "export": "Saving the trained model to files so the API (Module 6) can load and use it.",
    # explanations
    "XAI (explainable AI)": "Showing WHY the model made a decision, not just the decision.",
    "global explanation": "What matters most for the model in general, across all customers.",
    "local explanation": "Why the model decided this for ONE specific customer.",
    "native importance": "Importance read from inside the model itself: the size of the coefficients for logistic "
                         "regression, or how much each variable improves the splits for trees.",
    "permutation importance": "Shuffle one variable at random and measure how much worse the model gets. "
                              "Big drop = important variable. Works the same for every model, so it is comparable.",
    "SHAP": "Splits a prediction fairly between all the variables, like splitting a restaurant bill by what each "
            "person ordered. Positive = pushes towards churn, negative = pushes towards staying.",
    "SHAP value": "How much one variable moved this customer's churn probability up (+) or down (−).",
    "beeswarm": "A SHAP chart for many customers: one row per variable, one dot per customer.",
    "dependence plot": "How one variable's SHAP effect changes with its value (e.g. longer tenure → lower risk).",
    "waterfall": "A SHAP chart for ONE customer: bars showing each variable's push up or down.",
    "LIME": "A second explanation method: it slightly changes the customer's information many times, watches how "
            "the prediction changes, and fits a simple local model to explain it.",
    "SHAP vs LIME comparison": "Checking that both methods agree on which variables matter and in which direction, "
                               "like asking a second doctor for an opinion.",
    "rank correlation (Spearman)": "How similar two rankings are: 1 = identical order, 0 = unrelated.",
    "synthesis sentence": "An automatic plain-language sentence summarising the customer's risk and its main reasons.",
    "API": "A separate small program that the app asks for predictions and explanations, like a waiter taking "
           "orders to the kitchen.",
    "sensitive variables": "Personal traits such as gender or age (SeniorCitizen). Using them can make decisions unfair.",
}


def enabled() -> bool:
    return st.session_state.get("show_help", True)


def sidebar():
    st.sidebar.toggle("Show explanations (English)", value=True, key="show_help",
                      help="Turn off to hide all the explanation boxes, e.g. during a demo.")
    if enabled():
        with st.sidebar.expander("📘 What are these settings?"):
            st.markdown(SIDEBAR)


def page(key: str):
    if enabled():
        with st.expander("📘 What this page does", expanded=True):
            st.markdown(PAGES[key])


def chart(key: str):
    if enabled():
        with st.expander("📊 How to read this", expanded=False):
            st.markdown(CHARTS[key])


def terms(keys: list[str]):
    if enabled():
        with st.expander("📖 Terms on this page", expanded=False):
            st.markdown("\n".join(f"- **{k}**: {GLOSSARY[k]}" for k in keys))


SIDEBAR = """
- **URL de l'API d'interprétation**: the address of the API program (Module 6) that computes the explanations
  in Module 5. Start it with `uvicorn api.main:app --port 8000`.
- **Modules**: the 5 steps of the app, to be done in order. Module 6 is the API itself.
"""

# --------------------------------------------------------------------------- pages
PAGES = {
    "m1": """
**Step 1: get to know the data.** A phone / internet company wants to know which customers are about to leave.

- Click **Télécharger automatiquement** to download the Telco dataset from Kaggle (or upload your own CSV).
- Choose the **target variable**: the column we want to predict (`Churn`).
- The page then describes the data: what it looks like, empty cells, how many customers left, and charts
  that show which information is linked to leaving.
""",
    "m2": """
**Step 2: prepare the data for the models.**

- Choose which columns to **exclude** (the customer ID is excluded by default: it is just a label).
- The app builds a **pipeline** that fills empty cells, turns categories into numbers (one-hot) and puts
  numbers on the same scale.
- It **splits** the customers: most to learn from, the rest hidden for the final test. **Stratify** keeps the
  same share of churners in both parts.
""",
    "m3": """
**Step 3: train the models.** Pick which of the 5 models to train (all by default) and their settings.

- **Manual hyperparameters:** set the knobs yourself.
- **RandomizedSearchCV:** let the computer try combinations and keep the best (slower).

The table lists each trained model and the settings it used. Their performance is compared in Module 4.
""",
    "m4": """
**Step 4: measure how good each model is** on the hidden test customers.

- The **threshold** slider sets the cut-off probability above which a customer is predicted to leave.
  All threshold-based numbers update live.
- Compare models with the table, the ROC and precision-recall curves, and the confusion matrix.
- **Export** the chosen model so the API can serve it for Module 5.
""",
    "m5": """
**Step 5: explain the model.** This is the heart of the project: understand WHY the model predicts what it does.

- **Explications globales:** which variables matter most overall, and in which direction.
- **Explications locales:** pick one customer (or type a fictional one) and see the probability, the decision,
  and the reasons, with two methods (SHAP and LIME) compared side by side, plus a plain-language sentence.

The explanations are computed by the API (Module 6), which must be running with an exported model.
""",
}

# --------------------------------------------------------------------------- charts & tables
CHARTS = {
    "m1_target": "The **target variable** is the column the model learns to predict. Keep `Churn` (did the customer leave?).",
    "m1_preview": """
The first 20 customers. Each row is a customer, each column a piece of information: profile (`gender`,
`SeniorCitizen`, `Partner`, `Dependents`), services (`PhoneService`, `InternetService`, `TechSupport`…),
contract and billing (`tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`) and the answer
`Churn` (1 = left).
""",
    "m1_describe": """
**Summary of every column** (one row per column):
- For numbers: `count`, `mean` (average), `std` (spread), `min` / `max`, `25% / 50% / 75%` (quartiles,
  50 % = median).
- For text columns: `unique` (number of different values), `top` (most frequent value), `freq` (how often).
- Example: `tenure` goes from 0 to 72 months.
""",
    "m1_missing": """
Columns with **empty cells** and how many. Only `TotalCharges` has 11 empty values: brand-new customers
(tenure 0) who have not been billed yet. They are filled automatically in Module 2 (**imputation**).
""",
    "m1_balance": """
How many customers stayed (0) and left (1), as a count and a percentage: about **73.5 % stayed, 26.5 % left**.
Because the groups are unequal, **accuracy** is misleading and we also look at PR-AUC, precision and recall.
""",
    "m1_explore": """
Pick any variable in the list:
- **Number** (e.g. tenure) → **histogram**: how many customers have each value, coloured by churn (overlapping
  bars). Where the churn colour dominates, leaving is more common.
- **Category** (e.g. Contract) → **bar chart** of the churn rate per category. Taller bar = more customers leave.
- **Example:** Contract = month-to-month ≈ 43 % churn, one year ≈ 11 %, two years ≈ 3 %.
""",
    "m1_corr": """
**Correlation heatmap** between the number columns (and Churn).
- Each square = correlation between the row and the column, from −1 (blue) to +1 (red); 0 = no link.
- **What to see:** `tenure` vs `Churn` ≈ −0.35 (longer customers leave less); `MonthlyCharges` vs `Churn`
  ≈ +0.19 (higher bills, slightly more churn); `tenure` and `TotalCharges` are strongly linked (≈ 0.83).
""",
    "m2_exclude": "Columns removed before training. `customerID` is excluded: it is a label, not information about behaviour.",
    "m2_options": """
- **Normaliser (StandardScaler):** put numbers on the same scale (important for logistic regression and k-NN).
- **Proportion du jeu de test:** share of customers hidden for the final test (0.20 = 20 %).
- **Stratifier:** keep 26.5 % churners in both parts.
""",
    "m2_result": """
The confirmation message gives the size of each part (e.g. 5,634 to learn, 1,409 to test) and how many
variables are numbers vs categories. The same preparation is built into every model's **pipeline**, so it is
learned on the training part only (no leakage).
""",
    "m3_settings": """
- **Modèles à entraîner:** the 5 models required by the brief; untick any you want to skip.
- **Optimiser (RandomizedSearchCV):** tick to let the computer search the best settings (scored by AUC-ROC,
  3-fold cross-validation). Untick to type them yourself in *Hyperparamètres manuels*.
""",
    "m3_table": """
One row per trained model with the **settings it actually used**. Empty cells mean the setting does not apply
to that model. Performance is in Module 4.
""",
    "m4_threshold": """
**Seuil de décision:** customers with a churn probability above it are predicted to leave.
Lower it → catch more churners (recall ↑) but more false alarms (precision ↓). Raise it → the opposite.
AUC-ROC and PR-AUC do not change with the threshold; precision, recall and F1 do.
""",
    "m4_table": """
**Comparison of all models on the test customers** (one row per model):
- **AUC-ROC** and **PR-AUC**: how well the model ranks churners above loyal customers (higher = better),
  independent of the threshold.
- **precision**, **rappel** (recall), **F1**: at the current threshold.
- **What to see:** logistic regression is the best (AUC ≈ 0.84): a simple, readable model is enough here.
""",
    "m4_roc": """
**ROC curves**, one line per model.
- Horizontal: share of loyal customers wrongly flagged. Vertical: share of churners caught.
- A good model's curve rises quickly to the top-left. The dashed diagonal is random guessing.
- The **AUC** (area under the curve) is in the legend.
""",
    "m4_pr": """
**Precision-recall curves**, one line per model.
- Moving right = catching more churners (recall). Height = share of flagged customers who really leave (precision).
- A line that stays high while going right is better. Useful because churners are the smaller group.
""",
    "m4_cm": """
**Confusion matrix** for the chosen model at the current threshold.
- Rows = what really happened, columns = what the model predicted.
- Top-left: loyal customers correctly left alone. Bottom-right: churners correctly caught.
- Top-right: **false alarms** (predicted churn, stayed). Bottom-left: **missed churners**.
- Darker blue = more customers in that box.
""",
    "m4_live": """
Precision, recall and F1 of the chosen model **at the current threshold**. Move the threshold slider at the
top of the page and watch them change.
""",
    "m4_export": """
Saves the chosen model, its threshold and some sample customers into the `models/` folder. Then restart the API
(`uvicorn api.main:app --port 8000`) so it loads the new model, and open Module 5.
""",
    "m5_native": """
**Native importance**: read from inside the model. For logistic regression it is the size of each coefficient;
for trees, how much each variable helps the splits. Longer bar = more important.
Variables with a `_` (e.g. `Contract_Two year`) are one-hot columns: one per category.
""",
    "m5_perm": """
**Permutation importance**: how much the model's score drops when one variable is shuffled at random.
Longer bar = the model relies more on it. Works the same for every model, so it is comparable between models.
""",
    "m5_beeswarm": """
**SHAP beeswarm**: one row per variable (most important on top), one dot per customer.
- **Horizontal position:** effect on the churn probability. Right = pushes towards leaving, left = towards staying.
- **Colour:** the variable's value for that customer. Red = high, blue = low.
- **Example:** for `tenure`, blue dots (new customers) on the right and red dots (long-time customers) on the
  left mean *new customers are more likely to leave*.
""",
    "m5_dependence": """
**SHAP dependence plot** for the chosen variable: each dot is a customer.
- Horizontal: the variable's value. Vertical: its SHAP effect (above 0 = pushes towards churn).
- **What to see:** the shape of the effect, e.g. for `tenure` the effect falls as tenure grows.
""",
    "m5_choose": """
- **Choisir un individu du jeu de test:** pick a real hidden test customer by its index.
- Untick it to fill a **fictional customer** by hand: categories are dropdown lists, numbers are number boxes.
Then click **Expliquer cette prédiction**.
""",
    "m5_result": """
- **Probabilité de churn:** the model's estimate that this customer leaves.
- **Décision (seuil):** Churn if the probability is above the threshold exported in Module 4.
- The blue box is the **synthesis sentence**: the risk level and the 3 variables that weigh the most, with their
  direction (*augmente* = increases the risk, *diminue* = decreases it).
""",
    "m5_waterfall": """
**SHAP chart for this customer** (the 10 strongest effects):
- **Red bars** push the probability up (towards leaving), **blue bars** push it down (towards staying).
- Longer bar = stronger effect. Variables are one-hot columns (e.g. `Contract_Month-to-month`).
""",
    "m5_lime": """
**LIME explanation for the same customer** (its 10 main rules):
- Each bar is a rule such as `tenure <= 9.00` or `Contract=Month-to-month`.
- **Red** = pushes towards leaving, **blue** = towards staying. Longer = stronger.
- LIME splits numbers into ranges, so its labels look different from SHAP's.
""",
    "m5_compare": """
**SHAP vs LIME comparison**, per original variable (one-hot columns and LIME rules regrouped):
- **Rang SHAP / Rang LIME:** 1 = most influential for that method. Empty LIME rank = not in LIME's top 10.
- **Même signe:** do both methods agree on the direction (towards leaving or staying)?
- The sentence above sums it up: how many of the top 5 variables are shared, how many signs agree, and the
  **rank correlation** (1 = same order). Strong agreement = we can trust the explanation more.
""",
}
