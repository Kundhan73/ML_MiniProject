"""Figures for the report and slides."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#c2410c"  # default / risk
CALM = "#2563eb"    # repay / model
PALETTE = ["#2563eb", "#0d9488", "#9333ea", "#ca8a04", "#c2410c"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.6, "legend.frameon": False,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def class_balance(y, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    counts = y.value_counts().sort_index()
    ax.bar(["Repaid (0)", "Defaulted (1)"], counts.values, color=[CALM, ACCENT], width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("Applicants")
    ax.set_title("Class balance")
    ax.grid(axis="x")
    _save(fig, path)


def roc_pr(y, preds: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for (name, p), col in zip(preds.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=col, lw=1.6, label=f"{name} ({roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=col, lw=1.6, label=f"{name} ({average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    a2.axhline(y.mean(), ls="--", color=MUTED, lw=1, label=f"No skill ({y.mean():.3f})")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test, AUC)")
    a2.set(xlabel="Recall (defaulters caught)", ylabel="Precision", title="Precision-recall (test, AP)")
    a1.legend(fontsize=8, loc="lower right")
    a2.legend(fontsize=8, loc="upper right")
    _save(fig, path)


def importance(imp: pd.Series, path):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    s = imp[::-1]
    ax.barh(s.index, s.values, color=CALM, height=0.65)
    ax.set_xlabel("Mean |contribution| to log-odds (SHAP, test set)")
    ax.set_title("What drives the LightGBM prediction")
    ax.grid(axis="y")
    _save(fig, path)


def threshold_curve(y, p, thr, path):
    pr, rc, t = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(t, rc[:-1], color=ACCENT, label="Recall")
    ax.plot(t, pr[:-1], color=CALM, label="Precision")
    ax.plot(t, f1[:-1], color=INK, lw=1.2, label="F1")
    ax.axvline(thr, color=MUTED, ls="--", lw=1)
    ax.text(thr + 0.01, 0.92, f"chosen cut-off {thr:.2f}", color=INK, fontsize=8)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.text(0.51, 0.80, "default 0.5", color=MUTED, fontsize=8)
    ax.set(xlim=(0, 0.8), ylim=(0, 1), xlabel="Decision threshold on P(default)",
           title="Threshold trade-off (test)")
    ax.legend(loc="center right")
    _save(fig, path)


def calibration(y, p, path):
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot([0, 0.6], [0, 0.6], ls="--", color=MUTED, lw=1)
    ax.plot(mp, fr, marker="o", color=CALM)
    ax.set(xlabel="Predicted P(default)", ylabel="Observed default rate", title="Calibration (deciles)")
    _save(fig, path)


def policy(y, p, path):
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    frac = np.arange(1, len(ys) + 1) / len(ys)
    caught = np.cumsum(ys) / ys.sum()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(frac, caught, color=CALM, label="LightGBM")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, label="Random")
    for r in (0.1, 0.2):
        c = caught[int(r * len(ys)) - 1]
        ax.scatter([r], [c], color=ACCENT, zorder=3)
        ax.text(r + 0.02, c - 0.06, f"reject {r:.0%} -> catch {c:.0%}", fontsize=8)
    ax.set(xlabel="Share of applicants rejected (riskiest first)", ylabel="Share of defaulters caught",
           title="Gain chart (test)")
    ax.legend(loc="lower right")
    _save(fig, path)


def paper_vs_ours(R, path):
    names = ["Logistic Regression", "Random Forest", "Naive Bayes", "MLP (64-32)", "LightGBM"]
    paper_keys = ["Logistic Regression", "Random Forest", "Naive Bayes", "Multi-layer Perceptron", "LightGBM"]
    paper = [R["paper_reported"][k] for k in paper_keys]
    ours = [R["paper_style"][n]["accuracy"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.18, paper, 0.36, color=MUTED, label="Paper (down-sampled data)")
    ax.bar(x + 0.18, ours, 0.36, color=CALM, label="Ours (balanced test)")
    for i, (a, b) in enumerate(zip(paper, ours)):
        ax.text(i - 0.18, a + 0.01, f"{a:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.18, b + 0.01, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, ["LogReg", "RF", "NB", "MLP", "LightGBM"])
    ax.set(ylim=(0, 0.85), ylabel="Accuracy on 50/50 test", title="Same evaluation as the paper")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x")
    _save(fig, path)


def pca_scatter(Z, y, clusters, path):
    rng = np.random.RandomState(0)
    idx = rng.choice(len(y), min(6000, len(y)), replace=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    for v, col, lab in ((0, CALM, "Repaid"), (1, ACCENT, "Defaulted")):
        m = idx[y[idx] == v]
        a1.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.35 if v == 0 else 0.6, color=col, label=lab)
    for c in np.unique(clusters):
        m = idx[clusters[idx] == c]
        a2.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.4, color=PALETTE[c % 5], label=f"Cluster {c + 1}")
    a1.set(title="PCA (2 of 20 comps), by outcome", xlabel="PC1", ylabel="PC2")
    a2.set(title="Same points, K-means (k=4)", xlabel="PC1")
    a1.legend(markerscale=4, fontsize=8)
    a2.legend(markerscale=4, fontsize=8)
    _save(fig, path)


def ext_source_effect(X, y, path):
    s = pd.qcut(X["EXT_SOURCE_MEAN"], 10, labels=False, duplicates="drop")
    rate = y.groupby(s).mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(rate.index + 1, rate.values, color=ACCENT, width=0.7)
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1)
    ax.set(xlabel="EXT_SOURCE_MEAN decile (1 = lowest score)", ylabel="Default rate",
           title="Default rate by external score")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x")
    _save(fig, path)
