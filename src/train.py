"""Run every experiment, write reports/results.json + figures, save the demo model.

    python -m src.train            # full run (~25 min on 2 CPU cores)
    python -m src.train --quick    # 20% sample, for a smoke test

All resampling and preprocessing is fitted inside each CV fold (imblearn Pipeline),
so nothing about the validation fold leaks into training.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import data as D
from src import plots as P

warnings.filterwarnings("ignore")
SEED = 42
REPORTS = D.ROOT / "reports"
MODELS = D.ROOT / "models"

LGBM_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
    random_state=SEED, n_jobs=2, verbose=-1,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- helpers
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cats = D.categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         # float32 halves memory, which matters once SMOTE adds synthetic rows
                         ("f32", FunctionTransformer(np.float32))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                         ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", dtype=np.float32,
                                                  min_frequency=50, sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)])


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
        threshold=thr, accuracy=accuracy_score(y, yhat),
        balanced_accuracy=balanced_accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat), f1=f1_score(y, yhat),
    )


def cv_oof(make_model, X, y, n_splits=5, lgbm_native=False):
    """Out-of-fold predictions. make_model() returns a fresh estimator/pipeline."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, va in skf.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
    return oof, fold_auc


CACHE = {}       # name -> (result, test predictions); persisted so a crashed run can resume
CACHE_PATH = None


def evaluate(name, make_model, Xtr, ytr, Xte, yte, n_splits=5):
    if name in CACHE:
        res, p = CACHE[name]
        log(f"{name:<34} (cached) test AUC {res['test']['roc_auc']:.4f}")
        return res, None, p
    t0 = time.time()
    oof, fold_auc = cv_oof(make_model, Xtr, ytr, n_splits)
    thr = best_f1_threshold(ytr, oof)  # chosen on train OOF only, never on test
    m = make_model().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    res = dict(
        cv_roc_auc_mean=float(np.mean(fold_auc)), cv_roc_auc_std=float(np.std(fold_auc)),
        cv_pr_auc=float(average_precision_score(ytr, oof)),
        test=metrics_at(yte, p, thr), test_at_0_5=metrics_at(yte, p, 0.5),
        seconds=round(time.time() - t0, 1),
    )
    log(f"{name:<34} CV AUC {res['cv_roc_auc_mean']:.4f}±{res['cv_roc_auc_std']:.4f} | "
        f"test AUC {res['test']['roc_auc']:.4f} PR-AUC {res['test']['pr_auc']:.4f} "
        f"R {res['test']['recall']:.3f} P {res['test']['precision']:.3f} ({res['seconds']}s)")
    CACHE[name] = (res, p)
    if CACHE_PATH is not None:
        pd.to_pickle(CACHE, CACHE_PATH)
    return res, m, p


# --------------------------------------------------------------------------- main
def main(quick=False):
    global CACHE, CACHE_PATH
    t_start = time.time()
    CACHE_PATH = REPORTS / ("cache_quick.pkl" if quick else "cache.pkl")
    if CACHE_PATH.exists():
        CACHE = pd.read_pickle(CACHE_PATH)
        log(f"resuming: {len(CACHE)} cached experiments")
    df = D.engineer(D.load_raw())
    if quick:
        df = df.sample(frac=0.2, random_state=SEED)
    X, y = D.split_xy(df)
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
        X, y, df[D.ID_COL], test_size=0.2, stratify=y, random_state=SEED)
    Xtr_l, cats = D.to_lgbm_frame(Xtr)
    Xte_l, _ = D.to_lgbm_frame(Xte, cats)
    pos = ytr.mean()
    spw = (1 - pos) / pos
    R = dict(dataset=dict(rows=len(df), features_after_engineering=X.shape[1],
                          default_rate=float(y.mean()), train_rows=len(ytr), test_rows=len(yte),
                          categorical=len(cats), missing_cells_pct=float(X.isna().mean().mean() * 100)))
    log(f"data {df.shape}, default rate {y.mean():.4f}, scale_pos_weight {spw:.2f}")

    # ---- 1. model comparison --------------------------------------------------
    pre = lambda: make_preprocessor(Xtr)
    models = {
        "Logistic Regression": lambda: Pipeline([("pre", pre()), ("clf", LogisticRegression(
            C=0.05, class_weight="balanced", max_iter=2000))]),
        "Naive Bayes": lambda: Pipeline([("pre", pre()), ("rus", RandomUnderSampler(random_state=SEED)),
                                         ("clf", GaussianNB())]),
        "Random Forest": lambda: Pipeline([("pre", pre()), ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=25, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2, random_state=SEED))]),
        "MLP (64-32)": lambda: Pipeline([("pre", pre()), ("rus", RandomUnderSampler(random_state=SEED)),
                                         ("clf", MLPClassifier((64, 32), alpha=1e-2, batch_size=256,
                                                               early_stopping=True, max_iter=200,
                                                               random_state=SEED))]),
    }
    R["models"], test_preds = {}, {}
    for name, mk in models.items():
        R["models"][name], _, test_preds[name] = evaluate(name, mk, Xtr, ytr, Xte, yte)
    name = "LightGBM"
    R["models"][name], final_model, test_preds[name] = evaluate(
        name, lambda: lgb.LGBMClassifier(**LGBM_PARAMS), Xtr_l, ytr, Xte_l, yte)

    # ---- 2. imbalance strategies ----------------------------------------------
    log("imbalance strategies")
    R["imbalance"] = {}
    strategies = {
        "LightGBM | none": lambda: lgb.LGBMClassifier(**LGBM_PARAMS),
        "LightGBM | class weights": lambda: lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw),
        "LightGBM | undersample": lambda: Pipeline([("rus", RandomUnderSampler(random_state=SEED)),
                                                     ("clf", lgb.LGBMClassifier(**LGBM_PARAMS))]),
    }
    for k, mk in strategies.items():
        R["imbalance"][k], _, _ = evaluate(k, mk, Xtr_l, ytr, Xte_l, yte, n_splits=3)
    lr = lambda: LogisticRegression(C=0.05, max_iter=2000)
    for k, steps in {
        "LogReg | none": [],
        "LogReg | undersample": [("rus", RandomUnderSampler(random_state=SEED))],
        # SMOTE to a 1:2 ratio (not 1:1) keeps the synthetic set small enough for 8 GB RAM
        "LogReg | SMOTE": [("smote", SMOTE(sampling_strategy=0.5, random_state=SEED))],
    }.items():
        mk = lambda steps=steps: Pipeline([("pre", pre()), *steps, ("clf", lr())])
        R["imbalance"][k], _, _ = evaluate(k, mk, Xtr, ytr, Xte, yte, n_splits=3)
    R["imbalance"]["LogReg | class weights"] = R["models"]["Logistic Regression"]

    # ---- 3. paper-style evaluation (balanced test set, accuracy) ----------------
    log("paper-style balanced evaluation")
    rng = np.random.RandomState(SEED)
    te_pos = np.where(yte == 1)[0]
    te_neg = rng.choice(np.where(yte == 0)[0], len(te_pos), replace=False)
    bal = np.concatenate([te_pos, te_neg])
    R["paper_style"] = {}
    for name, p in test_preds.items():
        yb, pb = yte.iloc[bal].values, p[bal]
        # Models trained on balanced data/weights: 0.5. Unweighted LightGBM: shift the cut-off
        # to the training prior, which is the equivalent decision rule for a 50/50 test set.
        thr = float(pos) if name == "LightGBM" else 0.5
        yhat = (pb >= thr).astype(int)
        R["paper_style"][name] = dict(
            accuracy=accuracy_score(yb, yhat),
            precision_repay=precision_score(yb, yhat, pos_label=0, zero_division=0),
            precision_default=precision_score(yb, yhat, pos_label=1, zero_division=0),
            recall_repay=recall_score(yb, yhat, pos_label=0), recall_default=recall_score(yb, yhat, pos_label=1),
            roc_auc=roc_auc_score(yb, pb))
    R["paper_reported"] = {"Logistic Regression": 0.6934, "Random Forest": 0.6351, "Naive Bayes": 0.5211,
                           "Multi-layer Perceptron": 0.6915, "LightGBM": 0.5747}

    # ---- 4. EXT_SOURCE ablation -------------------------------------------------
    log("EXT_SOURCE ablation")
    Xtr_n = Xtr_l.drop(columns=D.EXT_FEATURES)
    Xte_n = Xte_l.drop(columns=D.EXT_FEATURES)
    R["ablation_no_ext"], _, _ = evaluate("LightGBM without EXT_SOURCE", lambda: lgb.LGBMClassifier(**LGBM_PARAMS),
                                          Xtr_n, ytr, Xte_n, yte, n_splits=3)
    no_ext_rows = Xte["EXT_SOURCE_N_MISSING"] >= 2
    R["thin_file"] = dict(
        share_test=float(no_ext_rows.mean()),
        default_rate=float(yte[no_ext_rows].mean()),
        auc_full_model=roc_auc_score(yte[no_ext_rows], test_preds["LightGBM"][no_ext_rows.values]),
    )

    # ---- 4b. what the history tables add --------------------------------------------
    hist_cols = D.history_columns(Xtr_l.columns)
    if hist_cols:
        log(f"history ablation ({len(hist_cols)} history features)")
        app_cols = [c for c in Xtr_l.columns if c not in hist_cols]
        R["ablation_app_only"], _, _ = evaluate("LightGBM application table only",
                                                lambda: lgb.LGBMClassifier(**LGBM_PARAMS),
                                                Xtr_l[app_cols], ytr, Xte_l[app_cols], yte, n_splits=3)
        no_bureau = (Xte["HIST_HAS_BUREAU"] == 0).values
        R["no_bureau_history"] = dict(
            share_test=float(no_bureau.mean()), default_rate=float(yte[no_bureau].mean()),
            auc_full_model=roc_auc_score(yte[no_bureau], test_preds["LightGBM"][no_bureau]))
        R["history_features"] = len(hist_cols)

    # ---- 5. K-means + per-cluster models (paper section V.C) ---------------------
    log("k-means per-cluster replication")
    pre_k = make_preprocessor(Xtr).fit(Xtr)
    Ztr = PCA(20, random_state=SEED).fit(pre_k.transform(Xtr))
    pca = Ztr
    Ztr, Zte = pca.transform(pre_k.transform(Xtr)), pca.transform(pre_k.transform(Xte))
    R["kmeans"] = {}
    for k in (2, 3, 4, 5):
        km = KMeans(k, n_init=5, random_state=SEED).fit(Ztr)
        ctr, cte = km.labels_, km.predict(Zte)
        p = np.zeros(len(yte))
        for c in range(k):
            m = lgb.LGBMClassifier(**LGBM_PARAMS).fit(Xtr_l[ctr == c], ytr[ctr == c])
            p[cte == c] = m.predict_proba(Xte_l[cte == c])[:, 1]
        R["kmeans"][k] = dict(test_roc_auc=roc_auc_score(yte, p), test_pr_auc=average_precision_score(yte, p),
                              cluster_sizes=np.bincount(ctr).tolist(),
                              cluster_default_rates=[float(ytr[ctr == c].mean()) for c in range(k)])
        log(f"  k={k}: per-cluster AUC {R['kmeans'][k]['test_roc_auc']:.4f}")
        if k == 4:
            km4_te = cte
    R["kmeans_global_auc"] = R["models"]["LightGBM"]["test"]["roc_auc"]
    R["pca_var_2"] = float(PCA(2, random_state=SEED).fit(pre_k.transform(Xtr)).explained_variance_ratio_.sum())

    # ---- 6. final model: importance, policy, artefacts ---------------------------
    log("final model analysis")
    p_final = test_preds["LightGBM"]
    thr = R["models"]["LightGBM"]["test"]["threshold"]
    if final_model is None:  # LightGBM result came from the cache: refit (seeded, so identical)
        final_model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(Xtr_l, ytr)
    contrib = final_model.predict(Xte_l, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=Xte_l.columns).sort_values(ascending=False)
    gain = pd.Series(final_model.booster_.feature_importance("gain"), index=Xte_l.columns)
    R["importance_top20"] = imp.head(20).round(5).to_dict()
    R["gain_top20"] = (gain / gain.sum()).sort_values(ascending=False).head(20).round(5).to_dict()

    order = np.argsort(-p_final)
    ysort = yte.values[order]
    R["policy"] = {}
    for rej in (0.05, 0.10, 0.20, 0.30):
        n = int(rej * len(ysort))
        R["policy"][f"reject_{int(rej*100)}pct"] = dict(
            defaulters_caught=float(ysort[:n].sum() / ysort.sum()),
            default_rate_approved=float(ysort[n:].mean()), baseline_default_rate=float(yte.mean()))

    # artefacts for the demo (a --quick run writes to reports/quick/ so it never
    # overwrites the committed model, figures or demo sample)
    out = REPORTS / "quick" if quick else None
    models_dir = out / "models" if quick else MODELS
    figs = out / "figures" if quick else REPORTS / "figures"
    sample_path = (out if quick else D.ROOT / "app") / "sample_applicants.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(models_dir / "lgbm_model.txt"))
    raw_all = D.load_raw().set_index(D.ID_COL)
    raw_cols = [c for c in raw_all.columns if c != D.TARGET]
    raw_tr = raw_all.loc[idtr.values]
    defaults = {c: (raw_tr[c].mode().iloc[0] if not pd.api.types.is_numeric_dtype(raw_tr[c])
                    else float(raw_tr[c].median())) for c in raw_cols}
    choices = {c: sorted(raw_tr[c].dropna().unique().tolist()) for c in raw_cols
               if not pd.api.types.is_numeric_dtype(raw_tr[c])}
    meta = dict(features=list(Xtr_l.columns), categories=cats, threshold=thr, defaults=defaults,
                choices=choices, raw_columns=raw_cols, test_roc_auc=R["models"]["LightGBM"]["test"]["roc_auc"],
                history_columns=D.history_columns(raw_cols))
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=1, default=float))
    raw_te = raw_all.loc[idte.values].reset_index()
    sample = raw_te.sample(300, random_state=SEED)
    sample.to_csv(sample_path, index=False)

    # ---- figures ------------------------------------------------------------------
    log("figures")
    P.class_balance(y, figs / "class_balance.png")
    P.roc_pr(yte, test_preds, figs / "roc_pr.png")
    P.importance(imp.head(15), figs / "importance.png")
    P.threshold_curve(yte, p_final, thr, figs / "threshold.png")
    P.calibration(yte, p_final, figs / "calibration.png")
    P.policy(yte, p_final, figs / "policy.png")
    P.paper_vs_ours(R, figs / "paper_vs_ours.png")
    P.pca_scatter(Zte[:, :2], yte.values, km4_te, figs / "pca_kmeans.png")
    P.ext_source_effect(Xte, yte, figs / "ext_source.png")

    R["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    ((out / "results.json") if quick else (REPORTS / "results.json")).write_text(
        json.dumps(R, indent=2, default=float))
    log(f"done in {R['runtime_minutes']} min")
    CACHE_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
