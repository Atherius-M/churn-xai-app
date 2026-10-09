from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=4)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1a3c6e"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=12)
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=11)

doc = SimpleDocTemplate(
    "one_pager.pdf", pagesize=A4,
    topMargin=14 * mm, bottomMargin=14 * mm, leftMargin=16 * mm, rightMargin=16 * mm,
)

story = []
story.append(Paragraph("Churn XAI — Classification Explicable de l'Attrition Client", title_style))
story.append(Paragraph(
    "Projet Challenge 1 — Data Mining, ENSI 2026-2027 — Cas métier : Télécom, Telco Customer Churn (IBM/Kaggle)",
    subtitle_style,
))

story.append(Paragraph("Problème métier", h2))
story.append(Paragraph(
    "Un opérateur télécom souhaite anticiper les résiliations (churn) pour cibler des actions de "
    "rétention avant la perte du client. Le modèle doit être performant et explicable : un conseiller "
    "doit comprendre pourquoi un client donné est jugé à risque pour choisir le bon levier d'action.",
    body,
))

story.append(Paragraph("Données", h2))
story.append(Paragraph(
    "Telco Customer Churn (IBM/Kaggle) : 7 043 clients, 20 variables explicatives (contrat, services "
    "souscrits, ancienneté, facturation...) + cible binaire <b>Churn</b> (26,5% de résiliations). "
    "Téléchargement automatisé via <i>kagglehub</i>, avec repli sur import CSV manuel.",
    body,
))

story.append(Paragraph("Meilleur modèle et résultats", h2))
story.append(Paragraph(
    "5 modèles comparés (régression logistique, arbre de décision, k-NN, Random Forest, XGBoost) sur "
    "un split 80/20 stratifié. La <b>régression logistique</b> obtient le meilleur AUC-ROC, à "
    "performance quasi égale avec XGBoost — un modèle simple et interprétable suffit sur ce cas.",
    body,
))

table_data = [
    ["Modèle", "AUC-ROC", "PR-AUC", "Précision", "Rappel", "F1"],
    ["Régression logistique", "0.842", "0.632", "0.657", "0.559", "0.604"],
    ["XGBoost", "0.833", "0.640", "0.630", "0.505", "0.561"],
    ["Arbre de décision", "0.830", "0.648", "0.535", "0.567", "0.599"],
    ["K plus proches voisins", "0.822", "0.613", "0.585", "0.564", "0.574"],
    ["Random Forest", "0.817", "0.606", "0.607", "0.479", "0.535"],
]
t = Table(table_data, colWidths=[42 * mm, 20 * mm, 20 * mm, 22 * mm, 18 * mm, 15 * mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dce6f2")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
story.append(Spacer(1, 6))

story.append(Paragraph("Explications clés (SHAP / LIME)", h2))
story.append(Paragraph(
    "Globalement, les trois variables dominantes (importance native, permutation, et SHAP) sont "
    "<b>tenure</b> (ancienneté), <b>Contract</b> (type de contrat) et <b>InternetService</b> "
    "(type de connexion internet) — un client récent, en contrat mensuel, avec fibre optique, "
    "présente le risque de churn le plus élevé. Sur les individus testés, SHAP et LIME s'accordent "
    "sur le signe et l'ordre de ces variables, ce qui renforce la confiance dans l'explication locale. "
    "Exemple de phrase de synthèse générée automatiquement : <i>&laquo;Ce client a risque de depart "
    "faible (probabilite 4.5%), principalement a cause de : tenure (diminue le risque), TotalCharges "
    "(augmente le risque), MonthlyCharges (diminue le risque).&raquo;</i>",
    body,
))

story.append(Paragraph("Recommandations et limites", h2))
items = [
    "Cibler en priorité les clients à faible ancienneté, en contrat mensuel et abonnés à la fibre "
    "optique pour des offres de fidélisation (passage à un contrat annuel, remise les premiers mois).",
    "Le modèle utilise <b>gender</b> et <b>SeniorCitizen</b> ; leur importance reste faible comparée à "
    "tenure/Contract, mais un déploiement réel devrait auditer l'équité du score par sous-groupe avant "
    "mise en production.",
    "Le jeu de données est un échantillon figé (pas de dérive temporelle testée) : les coefficients et "
    "importances devront être réévalués périodiquement en production.",
    "La régression logistique étant quasi aussi performante que XGBoost ici, privilégier le modèle "
    "interprétable pour la mise en production tant que l'écart de performance reste négligeable.",
]
story.append(ListFlowable(
    [ListItem(Paragraph(it, body), bulletColor=colors.HexColor("#1a3c6e")) for it in items],
    bulletType="bullet", start="circle", leftIndent=10,
))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "Code source, notebook et API : voir le dépôt GitHub du projet (README.md).",
    small,
))

doc.build(story)
print("one_pager.pdf generated")
