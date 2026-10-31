"""Loanliness demo: score a loan applicant and explain the score.

    streamlit run app/app.py

Inference only. Needs models/lgbm_model.txt and models/metadata.json (both committed),
so it runs on any laptop without the training data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402

st.set_page_config(page_title="Loanliness: repayment risk", layout="wide")


@st.cache_resource
def load():
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_model.txt"))
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    sample = pd.read_csv(ROOT / "app" / "sample_applicants.csv")
    return booster, meta, sample


booster, meta, sample = load()


def score(raw: dict) -> tuple[float, pd.Series]:
    vals = {k: (np.nan if v is None else v) for k, v in raw.items()}
    for k, v in vals.items():
        if k not in meta["choices"]:
            vals[k] = pd.to_numeric(v, errors="coerce")
    vals[D.TARGET] = 0
    vals[D.ID_COL] = 0
    row = pd.DataFrame([vals])
    num = [c for c in row.columns if c not in meta["choices"]]
    row[num] = row[num].astype(float)
    X, _ = D.split_xy(D.engineer(row))
    X, _ = D.to_lgbm_frame(X, meta["categories"])
    X = X[meta["features"]]
    p = float(booster.predict(X)[0])
    contrib = booster.predict(X, pred_contrib=True)[0][:-1]
    return p, pd.Series(contrib, index=meta["features"])


st.title("Loanliness: will this applicant repay?")
st.caption(f"LightGBM trained on Home Credit applications · test ROC-AUC {meta['test_roc_auc']:.3f}. "
           "Coursework demo, not a lending tool.")

mode = st.radio("Applicant", ["Pick a real applicant from the held-out test set", "Enter details by hand"],
                horizontal=True)
raw = dict(meta["defaults"])
truth = None

if mode.startswith("Pick"):
    c1, c2 = st.columns([3, 1])
    filt = c2.selectbox("Show", ["All", "Only defaulters", "Only repayers"])
    pool = sample if filt == "All" else sample[sample[D.TARGET] == (1 if filt == "Only defaulters" else 0)]
    sid = c1.selectbox("Applicant ID", pool[D.ID_COL].tolist())
    rec = pool[pool[D.ID_COL] == sid].iloc[0]
    truth = int(rec[D.TARGET])
    raw.update({k: (None if pd.isna(v) else v) for k, v in rec.items() if k in meta["raw_columns"]})
else:
    ch = meta["choices"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Person")
        age = st.slider("Age", 20, 70, 40)
        raw["DAYS_BIRTH"] = -age * 365.25
        pens = st.checkbox("Pensioner / not employed")
        yrs = st.slider("Years in current job", 0.0, 40.0, 5.0, 0.5, disabled=pens)
        raw["DAYS_EMPLOYED"] = D.DAYS_EMPLOYED_SENTINEL if pens else -yrs * 365.25
        raw["CODE_GENDER"] = st.selectbox("Gender", ["F", "M"])
        raw["NAME_EDUCATION_TYPE"] = st.selectbox("Education", ch["NAME_EDUCATION_TYPE"],
                                                  index=ch["NAME_EDUCATION_TYPE"].index(meta["defaults"]["NAME_EDUCATION_TYPE"]))
        raw["NAME_FAMILY_STATUS"] = st.selectbox("Family status", ch["NAME_FAMILY_STATUS"],
                                                 index=ch["NAME_FAMILY_STATUS"].index(meta["defaults"]["NAME_FAMILY_STATUS"]))
        raw["CNT_CHILDREN"] = st.number_input("Children", 0, 10, 0)
        raw["CNT_FAM_MEMBERS"] = raw["CNT_CHILDREN"] + st.number_input("Adults in household", 1, 5, 2)
    with c2:
        st.subheader("Money")
        raw["NAME_INCOME_TYPE"] = st.selectbox("Income type", ch["NAME_INCOME_TYPE"],
                                               index=ch["NAME_INCOME_TYPE"].index(meta["defaults"]["NAME_INCOME_TYPE"]))
        raw["AMT_INCOME_TOTAL"] = st.number_input("Annual income", 10000, 5_000_000, 150_000, 10_000)
        raw["NAME_CONTRACT_TYPE"] = st.selectbox("Loan type", ch["NAME_CONTRACT_TYPE"])
        raw["AMT_CREDIT"] = st.number_input("Loan amount", 10000, 5_000_000, 500_000, 10_000)
        raw["AMT_GOODS_PRICE"] = st.number_input("Price of goods financed", 0, 5_000_000, 450_000, 10_000)
        raw["AMT_ANNUITY"] = st.number_input("Yearly repayment (annuity)", 1000, 500_000, 25_000, 1000)
        raw["FLAG_OWN_CAR"] = "Y" if st.checkbox("Owns a car") else "N"
        raw["FLAG_OWN_REALTY"] = "Y" if st.checkbox("Owns property", True) else "N"
    with c3:
        st.subheader("Credit history")
        st.caption("EXT_SOURCE_1-3 are normalised scores from external credit bureaus (0 = worst, 1 = best). "
                   "Untick to simulate a thin-file applicant with no score.")
        for i, dflt in ((1, 0.5), (2, 0.55), (3, 0.5)):
            if st.checkbox(f"Has EXT_SOURCE_{i}", i != 1):
                raw[f"EXT_SOURCE_{i}"] = st.slider(f"EXT_SOURCE_{i}", 0.0, 1.0, dflt, 0.01)
            else:
                raw[f"EXT_SOURCE_{i}"] = np.nan
        raw["REGION_RATING_CLIENT_W_CITY"] = st.select_slider("Region rating (1 best, 3 worst)", [1, 2, 3], 2)
        raw["REGION_RATING_CLIENT"] = raw["REGION_RATING_CLIENT_W_CITY"]
        hist = meta.get("history_columns", [])
        if hist:
            st.markdown("**Past loans** (bureau records and earlier Home Credit loans)")
            if st.checkbox("Has a repayment history", True):
                late = st.slider("Share of past instalments paid late", 0.0, 1.0, 0.05, 0.01)
                dpd = st.slider("Average days late on a late payment", 0, 60, 5)
                for c in ("INST_LATE_MEAN", "INST_1Y_LATE_MEAN", "INST_6M_LATE_MEAN"):
                    raw[c] = late
                for c in ("INST_DPD_MEAN", "INST_1Y_DPD_MEAN"):
                    raw[c] = late * dpd
                raw["INST_DPD_MAX"] = dpd * 2 if late > 0 else 0
                raw["INST_LATE_30_MEAN"] = late if dpd > 30 else 0.0
                raw["POS_SK_DPD_MEAN"] = late * dpd / 2
                raw["PREV_REFUSED_SUM"] = st.number_input("Earlier applications refused", 0, 20, 0)
                n_prev = max(raw.get("PREV_APPROVED_COUNT") or 1, raw["PREV_REFUSED_SUM"])
                raw["PREV_REFUSED_MEAN"] = raw["PREV_REFUSED_SUM"] / n_prev
                raw["BUR_ACTIVE_SUM"] = st.number_input("Active loans at other lenders", 0, 30, 1)
            else:
                for c in hist:
                    raw[c] = np.nan
    st.caption("Fields not shown are set to the training-set median or most common value.")

p, contrib = score(raw)
thr = st.slider("Decision threshold on P(default)", 0.05, 0.60, float(round(meta["threshold"], 2)), 0.01,
                help=f"Default {meta['threshold']:.2f} = the F1-optimal cut-off found by cross-validation on the training set.")

left, right = st.columns([1, 2])
with left:
    st.metric("Probability of default", f"{p:.1%}", help="Base rate in the data is about 8%.")
    if p >= thr:
        st.error(f"High risk: flag for manual review (P ≥ {thr:.2f})")
    else:
        st.success(f"Low risk: approve (P < {thr:.2f})")
    if truth is not None:
        st.info(f"Actual outcome: {'DEFAULTED' if truth else 'repaid'}")
with right:
    st.markdown("**Why: the 10 features that moved this score most** (red raises risk, blue lowers it)")
    import matplotlib.pyplot as plt

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index, top.values, color=["#c2410c" if v > 0 else "#2563eb" for v in top.values])
    ax.axvline(0, color="#7b8794", lw=0.8)
    ax.set_xlabel("Contribution to log-odds of default")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    st.pyplot(fig, clear_figure=True)
