"""Per-applicant features from Home Credit's six history tables.

    python -m src.history      # reads data/raw/kaggle/*.csv, writes data/processed/history_features.pkl

Each table is aggregated to one row per applicant (SK_ID_CURR). Only past behaviour
is summarised and TARGET is never used, so these features cannot leak the label
and are safe to compute before the train/test split.

Prefixes: BUR_ (credit bureau), PREV_ (previous Home Credit applications),
INST_ (instalment payments), POS_ (POS/cash loan balances), CC_ (credit cards).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data import KAGGLE_DIR, PROCESSED_DIR

ID = "SK_ID_CURR"


def _read(name, usecols, **kw):
    t = time.time()
    df = pd.read_csv(KAGGLE_DIR / f"{name}.csv", usecols=usecols, **kw)
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    print(f"  read {name:<24} {len(df):>11,} rows in {time.time() - t:.0f}s", flush=True)
    return df


def _flatten(agg: pd.DataFrame, prefix: str) -> pd.DataFrame:
    agg.columns = [f"{prefix}{a}_{b.upper()}" for a, b in agg.columns]
    return agg


def bureau() -> pd.DataFrame:
    bb = _read("bureau_balance", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])
    # STATUS 1-5 = 1-30 ... 120+ days past due that month
    bb["LATE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("float32")
    bb["LATE_60"] = bb["STATUS"].isin(["2", "3", "4", "5"]).astype("float32")
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(BB_MONTHS=("MONTHS_BALANCE", "size"),
                                            BB_LATE_RATE=("LATE", "mean"),
                                            BB_LATE60_RATE=("LATE_60", "mean"))
    del bb

    b = _read("bureau", [ID, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
                         "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE",
                         "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
                         "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY"])
    b = b.join(bb_agg, on="SK_ID_BUREAU")
    b["ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype("float32")
    b["BAD_DEBT"] = (b["CREDIT_ACTIVE"] == "Bad debt").astype("float32")
    b["CREDIT_CARD"] = (b["CREDIT_TYPE"] == "Credit card").astype("float32")
    b["DEBT_CREDIT_RATIO"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"].replace(0, np.nan)
    b["HAS_OVERDUE"] = (b["AMT_CREDIT_SUM_OVERDUE"] > 0).astype("float32")

    agg = _flatten(b.groupby(ID).agg({
        "SK_ID_BUREAU": ["count"],
        "ACTIVE": ["sum", "mean"], "BAD_DEBT": ["sum"], "CREDIT_CARD": ["mean"],
        "DAYS_CREDIT": ["min", "max", "mean"], "DAYS_CREDIT_ENDDATE": ["max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"], "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"], "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"], "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"], "AMT_ANNUITY": ["sum"], "HAS_OVERDUE": ["mean"],
        "DEBT_CREDIT_RATIO": ["mean", "max"], "BB_MONTHS": ["sum"],
        "BB_LATE_RATE": ["mean", "max"], "BB_LATE60_RATE": ["mean", "max"],
    }), "BUR_")
    act = b[b["ACTIVE"] == 1].groupby(ID).agg(BUR_ACTIVE_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
                                              BUR_ACTIVE_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
                                              BUR_ACTIVE_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"))
    agg = agg.join(act)
    agg["BUR_ACTIVE_DEBT_RATIO"] = agg["BUR_ACTIVE_DEBT_SUM"] / agg["BUR_ACTIVE_CREDIT_SUM"].replace(0, np.nan)
    return agg


def previous() -> pd.DataFrame:
    p = _read("previous_application", [ID, "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "AMT_ANNUITY",
                                       "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
                                       "AMT_GOODS_PRICE", "DAYS_DECISION", "CNT_PAYMENT",
                                       "NAME_YIELD_GROUP", "DAYS_FIRST_DUE", "DAYS_LAST_DUE",
                                       "DAYS_TERMINATION", "RATE_DOWN_PAYMENT", "CODE_REJECT_REASON"])
    for c in ["DAYS_FIRST_DUE", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p.loc[p[c] == 365243, c] = np.nan  # same placeholder as in the application table
    p["APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype("float32")
    p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype("float32")
    p["HIGH_YIELD"] = (p["NAME_YIELD_GROUP"] == "high").astype("float32")
    p["REJECT_SCOFR"] = (p["CODE_REJECT_REASON"] == "SCOFR").astype("float32")
    p["APP_CREDIT_RATIO"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"].replace(0, np.nan)
    p["CREDIT_TERM"] = p["AMT_ANNUITY"] / p["AMT_CREDIT"].replace(0, np.nan)
    p["STILL_OPEN"] = (p["DAYS_TERMINATION"].isna() & (p["APPROVED"] == 1)).astype("float32")

    agg = _flatten(p.groupby(ID).agg({
        "APPROVED": ["count", "sum", "mean"], "REFUSED": ["sum", "mean"], "HIGH_YIELD": ["mean"],
        "REJECT_SCOFR": ["sum"], "AMT_ANNUITY": ["mean", "max"], "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"], "AMT_DOWN_PAYMENT": ["mean"], "AMT_GOODS_PRICE": ["mean"],
        "APP_CREDIT_RATIO": ["mean", "min"], "CREDIT_TERM": ["mean"], "RATE_DOWN_PAYMENT": ["mean"],
        "DAYS_DECISION": ["max", "mean"], "CNT_PAYMENT": ["mean", "max"],
        "DAYS_LAST_DUE": ["max"], "STILL_OPEN": ["sum"],
    }), "PREV_")
    last = p.sort_values("DAYS_DECISION").groupby(ID).tail(1).set_index(ID)
    agg["PREV_LAST_REFUSED"] = last["REFUSED"]
    agg["PREV_LAST_DAYS_DECISION"] = last["DAYS_DECISION"]
    recent = p[p["DAYS_DECISION"] >= -365].groupby(ID).agg(PREV_1Y_COUNT=("APPROVED", "size"),
                                                          PREV_1Y_REFUSED=("REFUSED", "sum"))
    return agg.join(recent)


def installments() -> pd.DataFrame:
    i = _read("installments_payments", [ID, "NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT",
                                        "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])
    i["DPD"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["DBD"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    i["LATE"] = (i["DPD"] > 0).astype("float32")
    i["LATE_30"] = (i["DPD"] > 30).astype("float32")
    i["PAY_RATIO"] = i["AMT_PAYMENT"] / i["AMT_INSTALMENT"].replace(0, np.nan)
    i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 1).astype("float32")
    i["SHORTFALL"] = (i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]).clip(lower=0)
    agg = _flatten(i.groupby(ID).agg({
        "DAYS_INSTALMENT": ["count"], "DPD": ["mean", "max", "sum"], "DBD": ["mean"],
        "LATE": ["mean", "sum"], "LATE_30": ["mean"], "PAY_RATIO": ["mean", "min"],
        "UNDERPAID": ["mean"], "SHORTFALL": ["sum", "max"], "AMT_PAYMENT": ["sum", "mean"],
        "AMT_INSTALMENT": ["sum"], "NUM_INSTALMENT_VERSION": ["nunique"],
    }), "INST_")
    for days, tag in ((365, "1Y"), (180, "6M")):
        r = i[i["DAYS_INSTALMENT"] >= -days].groupby(ID).agg(**{
            f"INST_{tag}_LATE_MEAN": ("LATE", "mean"), f"INST_{tag}_DPD_MEAN": ("DPD", "mean"),
            f"INST_{tag}_UNDERPAID_MEAN": ("UNDERPAID", "mean")})
        agg = agg.join(r)
    agg["INST_PAID_SHARE"] = agg["INST_AMT_PAYMENT_SUM"] / agg["INST_AMT_INSTALMENT_SUM"].replace(0, np.nan)
    return agg


def pos_cash() -> pd.DataFrame:
    p = _read("POS_CASH_balance", [ID, "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE",
                                   "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    p["COMPLETED"] = (p["NAME_CONTRACT_STATUS"] == "Completed").astype("float32")
    p["LATE"] = (p["SK_DPD"] > 0).astype("float32")
    agg = _flatten(p.groupby(ID).agg({
        "MONTHS_BALANCE": ["count", "min"], "CNT_INSTALMENT_FUTURE": ["mean"],
        "SK_DPD": ["mean", "max"], "SK_DPD_DEF": ["mean", "max"], "LATE": ["mean"], "COMPLETED": ["sum"],
    }), "POS_")
    last = p[p["MONTHS_BALANCE"] >= -3].groupby(ID).agg(POS_RECENT_FUTURE_INST=("CNT_INSTALMENT_FUTURE", "sum"),
                                                        POS_RECENT_DPD_MAX=("SK_DPD", "max"))
    return agg.join(last)


def credit_card() -> pd.DataFrame:
    c = _read("credit_card_balance", [ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
                                      "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
                                      "AMT_INST_MIN_REGULARITY", "CNT_DRAWINGS_CURRENT", "SK_DPD", "SK_DPD_DEF"])
    c["UTIL"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    c["PAID_MIN"] = (c["AMT_PAYMENT_TOTAL_CURRENT"] >= c["AMT_INST_MIN_REGULARITY"]).astype("float32")
    c["LATE"] = (c["SK_DPD"] > 0).astype("float32")
    agg = _flatten(c.groupby(ID).agg({
        "MONTHS_BALANCE": ["count"], "AMT_BALANCE": ["mean", "max"], "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "UTIL": ["mean", "max"], "AMT_DRAWINGS_CURRENT": ["mean", "sum"], "CNT_DRAWINGS_CURRENT": ["mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean"], "PAID_MIN": ["mean"], "SK_DPD": ["mean", "max"],
        "SK_DPD_DEF": ["max"], "LATE": ["mean"],
    }), "CC_")
    recent = c[c["MONTHS_BALANCE"] >= -6].groupby(ID).agg(CC_6M_UTIL_MEAN=("UTIL", "mean"),
                                                         CC_6M_LATE_MEAN=("LATE", "mean"))
    return agg.join(recent)


def build() -> pd.DataFrame:
    t = time.time()
    parts = []
    for name, fn in (("bureau", bureau), ("previous", previous), ("installments", installments),
                     ("pos_cash", pos_cash), ("credit_card", credit_card)):
        print(f"{name}:", flush=True)
        parts.append(fn())
        print(f"  -> {parts[-1].shape[1]} features for {len(parts[-1]):,} applicants", flush=True)
    feats = pd.concat(parts, axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan).astype("float32")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_pickle(PROCESSED_DIR / "history_features.pkl")
    print(f"wrote {feats.shape} in {(time.time() - t) / 60:.1f} min")
    return feats


if __name__ == "__main__":
    build()
