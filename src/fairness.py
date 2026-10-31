"""Gender check: how much does the model lean on CODE_GENDER, and what does removing it cost?

    python -m src.fairness      # writes reports/fairness.json (about 3 minutes)

Uses the same split, features, parameters and threshold rule as src/train.py.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import data as D
from src.train import LGBM_PARAMS, REPORTS, SEED, best_f1_threshold, cv_oof


def approval_by_gender(gender, p, thr):
    out = {}
    for g in ("F", "M"):
        m = (gender == g).values
        out[g] = dict(share=float(m.mean()), flagged_rate=float((p[m] >= thr).mean()))
    return out


def main():
    df = D.engineer(D.load_raw())
    X, y = D.split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    gender = Xte["CODE_GENDER"]
    Xtr_l, cats = D.to_lgbm_frame(Xtr)
    Xte_l, _ = D.to_lgbm_frame(Xte, cats)
    R = {"default_rate": {g: float(yte[(gender == g).values].mean()) for g in ("F", "M")}}
    for name, cols in (("with_gender", list(Xtr_l.columns)),
                       ("without_gender", [c for c in Xtr_l.columns if c != "CODE_GENDER"])):
        mk = lambda: lgb.LGBMClassifier(**LGBM_PARAMS)
        oof, _ = cv_oof(mk, Xtr_l[cols], ytr, n_splits=3)
        thr = best_f1_threshold(ytr, oof)
        p = mk().fit(Xtr_l[cols], ytr).predict_proba(Xte_l[cols])[:, 1]
        R[name] = dict(test_roc_auc=roc_auc_score(yte, p), test_pr_auc=average_precision_score(yte, p),
                       threshold=thr, flagged=approval_by_gender(gender, p, thr))
        print(name, json.dumps(R[name]), flush=True)
    (REPORTS / "fairness.json").write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
