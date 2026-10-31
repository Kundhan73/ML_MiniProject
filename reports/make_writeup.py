"""Build reports/writeup.pdf (two pages) from reports/results.json.

    python reports/make_writeup.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
FIG = HERE / "figures"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=9.6, leading=11.9,
                      alignment=TA_JUSTIFY, spaceAfter=3.5)
h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=10.2, leading=12,
                   spaceBefore=4, spaceAfter=2, alignment=0, textColor=colors.HexColor("#1f2933"))
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=13.5, leading=16, alignment=1,
                       spaceAfter=2)
sub = ParagraphStyle("s", parent=body, fontName="Times-Italic", fontSize=9.5, alignment=1, spaceAfter=6)
cap = ParagraphStyle("c", parent=body, fontName="Times-Italic", fontSize=8.3, leading=10, alignment=1)

M = R["models"]
lg = M["LightGBM"]
d = R["dataset"]
pol = R["policy"]
ps = R["paper_style"]
km = R["kmeans"]


def pct(x, n=1):
    return f"{x * 100:.{n}f}%"


F = json.loads((HERE / "fairness.json").read_text())
ao = R["ablation_app_only"]["test"]["roc_auc"]
nb = R["no_bureau_history"]

story = [
    Paragraph("Loanliness: Predicting Loan Repayment Ability with Machine Learning", title),
    Paragraph("UE24CS352A Machine Learning Mini-Project, PES University &nbsp;|&nbsp; "
              "Kundhan (PES1UG24AM327) and Tarun Gowda (PES1UG24AM303)", sub),

    Paragraph("1. Problem statement", h),
    Paragraph(
        "Lenders decide whether to approve a loan largely from credit history, which shuts out students, "
        "young workers and anyone else with a thin credit file. Following Liang, Jin and Wang (Stanford CS229), "
        "we frame repayment ability as binary classification: from an applicant's loan application and any "
        "past borrowing records, predict whether they will have payment difficulties (TARGET = 1). Because "
        "defaulters are rare, we judge models by how well they <i>rank</i> risk (ROC-AUC, PR-AUC) and by recall "
        "and precision at a chosen cut-off, not by accuracy. We also re-test three claims from the reference "
        "paper: that down-sampling gives the best models, that K-means clusters plus one model per cluster "
        "improve accuracy, and which features drive repayment.", body),

    Paragraph("2. Dataset", h),
    Paragraph(
        f"Kaggle's <i>Home Credit Default Risk</i> data: {d['rows']:,} labelled applicants. The main table has "
        f"120 attributes (income, loan amount and annuity, age, employment, family, housing, region, documents, "
        f"and three normalised external credit scores EXT_SOURCE_1-3). Six history tables add 58 million rows of "
        f"past behaviour: 1.7M loans reported to the credit bureau with 27M monthly statuses, 1.7M previous "
        f"Home Credit applications, 13.6M instalment payments, and monthly POS/cash and credit-card balances. "
        f"Only {pct(d['default_rate'])} of applicants defaulted, so predicting \"repays\" for everyone already "
        f"scores {pct(1 - d['default_rate'])} accuracy. We fixed one data error: DAYS_EMPLOYED (and three date "
        f"columns in previous applications) use 365,243 days, about 1,000 years, as a placeholder; it covers 18% "
        f"of applicants, mostly pensioners, and became a missing value plus a flag.", body),

    Paragraph("3. Approach", h),
    Paragraph(
        "<b>Features.</b> 13 affordability features from the application (credit/income, annuity/income, "
        "annuity/credit as repayment speed, goods price/credit, employment/age, income per person, and summaries "
        "of the external scores), plus 122 history features: each history table is aggregated to one row per "
        "applicant, for example share of instalments paid late (overall, last year, last 6 months), days past "
        "due, bureau debt/credit ratio, refused applications and credit-card utilisation. The label is never "
        f"used in these, so they cannot leak it. {d['features_after_engineering']} features in total. "
        f"<b>Protocol.</b> A stratified 80/20 split holds out {d['test_rows']:,} applicants, touched once at the "
        "end. Everything else uses 5-fold stratified cross-validation on the training part. Imputation, scaling, "
        "one-hot encoding and any resampling sit inside an imblearn Pipeline and are re-fitted on each training "
        "fold, so validation data never leaks into training (the paper does not say how it avoided this). The "
        "decision threshold maximises F1 on out-of-fold predictions and is then frozen. "
        "<b>Models.</b> The paper's five families: logistic regression (class-weighted), Gaussian Naive Bayes, "
        "random forest (class-weighted), a 64-32 MLP (under-sampled folds) and LightGBM. <b>Extra "
        "experiments:</b> four imbalance strategies, the paper's 50/50 evaluation, K-means (k = 2-5 on 20 PCA "
        "components) with one LightGBM per cluster, and ablations without history, external scores or gender.",
        body),

    Paragraph("4. Implementation and results", h),
]

rows = [["Model", "CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Recall", "Precision", "F1"]]
for name in ["LightGBM", "Logistic Regression", "Random Forest", "MLP (64-32)", "Naive Bayes"]:
    v = M[name]
    t = v["test"]
    rows.append([name, f"{v['cv_roc_auc_mean']:.3f} ± {v['cv_roc_auc_std']:.3f}", f"{t['roc_auc']:.3f}",
                 f"{t['pr_auc']:.3f}", f"{t['recall']:.2f}", f"{t['precision']:.2f}", f"{t['f1']:.2f}"])
tbl = Table(rows, colWidths=[36 * mm, 25 * mm, 23 * mm, 22 * mm, 16 * mm, 18 * mm, 13 * mm])
tbl.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3), ("FONT", (0, 1), (-1, -1), "Times-Roman", 8.8),
    ("FONT", (0, 1), (-1, 1), "Times-Bold", 8.8),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
]))

story += [
    Paragraph(
        "The code is a small Python package: <font face='Courier'>src/history.py</font> aggregates the history "
        "tables (about one minute), <font face='Courier'>src/data.py</font> loads, cleans and builds features, "
        "<font face='Courier'>src/train.py</font> runs every experiment and writes all numbers to "
        "<font face='Courier'>reports/results.json</font>, and a Streamlit demo, "
        "<font face='Courier'>app/app.py</font>, scores an applicant, applies an adjustable threshold and shows "
        "the ten features that moved that applicant's score (per-row SHAP values from LightGBM). "
        f"The full run takes {R['runtime_minutes']:.0f} minutes on two CPU cores. Table 1 gives held-out results.",
        body),
    tbl,
    Paragraph(f"Table 1. Held-out test set ({d['test_rows']:,} applicants). Recall, precision and F1 at the "
              "cut-off chosen by cross-validation.", cap),
    Spacer(1, 3),
    Table([[Image(str(FIG / "roc_pr.png"), width=118 * mm, height=49 * mm),
            Image(str(FIG / "importance.png"), width=62 * mm, height=45 * mm)]],
          colWidths=[120 * mm, 64 * mm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    Paragraph("Figure 1. Left: ROC and precision-recall curves. Right: mean absolute SHAP contribution of the top "
              "15 features in LightGBM.", cap),
    Spacer(1, 2),
    Paragraph(
        f"<b>LightGBM is the best model</b> (test ROC-AUC {lg['test']['roc_auc']:.3f}, PR-AUC "
        f"{lg['test']['pr_auc']:.3f}, {lg['test']['pr_auc'] / d['default_rate']:.1f} times the no-skill "
        f"{d['default_rate']:.3f}), ahead of logistic regression ({M['Logistic Regression']['test']['roc_auc']:.3f}) "
        f"and random forest ({M['Random Forest']['test']['roc_auc']:.3f}). Naive Bayes is near chance: its "
        f"independence assumption fails on hundreds of correlated features. The cross-validation spread for the top four models is "
        f"±0.004 or less, so these gaps are real. <b>History matters.</b> The same LightGBM on the application table "
        f"alone scores {ao:.3f}; the history features add {lg['test']['roc_auc'] - ao:.3f}. "
        f"<b>The threshold matters more than the algorithm.</b> At the default 0.5 cut-off LightGBM shows "
        f"{pct(lg['test_at_0_5']['accuracy'])} accuracy but catches only {pct(lg['test_at_0_5']['recall'])} of "
        f"defaulters; at the tuned cut-off of {lg['test']['threshold']:.2f} it catches "
        f"{pct(lg['test']['recall'], 0)}. "
        f"<b>Resampling does not improve the model.</b> Class weights, under-sampling and SMOTE lower ROC-AUC by "
        f"0.0005 to 0.008; they only move the effective threshold, which tuning does for free. This contradicts "
        f"the paper's finding that down-sampled data gives the best performance. "
        f"<b>The paper's evaluation, reproduced.</b> The paper reports accuracy on down-sampled 50/50 data. On a "
        f"50/50 test set every one of our models beats its counterpart: LightGBM "
        f"{pct(ps['LightGBM']['accuracy'])} (paper 57.5%), logistic regression "
        f"{pct(ps['Logistic Regression']['accuracy'])} (69.3%), random forest {pct(ps['Random Forest']['accuracy'])} "
        f"(63.5%), MLP {pct(ps['MLP (64-32)']['accuracy'])} (69.2%), Naive Bayes "
        f"{pct(ps['Naive Bayes']['accuracy'])} (52.1%).", body),
    Paragraph(
        f"<b>K-means does not help.</b> One LightGBM per cluster scores {km['2']['test_roc_auc']:.3f}, "
        f"{km['3']['test_roc_auc']:.3f}, {km['4']['test_roc_auc']:.3f} and {km['5']['test_roc_auc']:.3f} AUC for "
        f"k = 2 to 5, each below the single global model ({R['kmeans_global_auc']:.3f}): splitting leaves each "
        f"model fewer examples, and boosted trees already form their own segments. "
        f"<b>What drives risk.</b> The external scores dominate (default rate falls from 23% in their lowest "
        f"decile to 2% in the highest), followed by gender, the largest bureau debt/credit ratio, goods/credit, "
        f"employer type, annuity and repayment speed. Without the external scores AUC is still "
        f"{R['ablation_no_ext']['test']['roc_auc']:.3f}. The model also holds up for the people the paper set "
        f"out to help: {R['thin_file']['auc_full_model']:.3f} AUC for applicants missing two or more external "
        f"scores ({pct(R['thin_file']['share_test'], 0)} of test) and {nb['auc_full_model']:.3f} for those with "
        f"no credit-bureau record ({pct(nb['share_test'], 0)}). "
        f"<b>Gender.</b> Dropping CODE_GENDER costs only {F['with_gender']['test_roc_auc'] - F['without_gender']['test_roc_auc']:.4f} AUC "
        f"and narrows the share of men flagged as high risk from {pct(F['with_gender']['flagged']['M']['flagged_rate'])} to "
        f"{pct(F['without_gender']['flagged']['M']['flagged_rate'])} (women: {pct(F['with_gender']['flagged']['F']['flagged_rate'])} "
        f"to {pct(F['without_gender']['flagged']['F']['flagged_rate'])}). "
        f"<b>As a policy:</b> reviewing the riskiest 10% of applicants catches "
        f"{pct(pol['reject_10pct']['defaulters_caught'], 0)} of defaulters and cuts the default rate among "
        f"approved loans from {pct(pol['reject_10pct']['baseline_default_rate'])} to "
        f"{pct(pol['reject_10pct']['default_rate_approved'])}; reviewing 20% catches "
        f"{pct(pol['reject_20pct']['defaulters_caught'], 0)} and leaves {pct(pol['reject_20pct']['default_rate_approved'])}.",
        body),

    Spacer(1, 3),
    Table([[Image(str(FIG / "paper_vs_ours.png"), width=88 * mm, height=44 * mm),
            Image(str(FIG / "policy.png"), width=64 * mm, height=45 * mm)]],
          colWidths=[92 * mm, 70 * mm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    Paragraph("Figure 2. Left: accuracy on a 50/50 test set, paper vs ours. Right: share of defaulters caught when "
              "reviewing the riskiest applicants first.", cap),
    Paragraph("5. Conclusions", h),
    Paragraph(
        f"Gradient-boosted trees on application data, affordability ratios and aggregated repayment history give "
        f"the strongest and most stable model (ROC-AUC {lg['test']['roc_auc']:.3f}), in line with strong single "
        f"models on this benchmark. History was the largest single improvement. On data this imbalanced, the "
        f"metric and the threshold matter more than the algorithm, and resampling and clustering, both central to "
        f"the reference paper, added nothing once evaluation was leak-free. The model stays accurate for "
        f"applicants without bureau records or external scores. A deployable version should drop gender (it "
        f"costs almost nothing), audit approval rates across groups, calibrate the probabilities, and set the "
        f"threshold from the lender's cost of a default versus a lost customer rather than from F1.",
        body),
]

doc = SimpleDocTemplate(str(HERE / "writeup.pdf"), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=12 * mm, bottomMargin=12 * mm,
                        title="Loanliness: Predicting Loan Repayment Ability", author="Kundhan and Tarun Gowda")
doc.build(story)
print("wrote", HERE / "writeup.pdf")
