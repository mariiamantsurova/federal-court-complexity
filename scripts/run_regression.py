#!/usr/bin/env python3
"""Run baseline regressions and save metrics to docs/step4_regression.json"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "step4_regression.json"


def make_design(df: pd.DataFrame, *, include_judge: bool, interaction: bool) -> pd.DataFrame:
    parts = [df[["complexity_index"]].astype(float)]
    parts.append(pd.get_dummies(df["nature_suit_top"], prefix="suit", dtype=float))
    if include_judge:
        j = pd.get_dummies(df["District_Judge"], prefix="judge", dtype=float)
        parts.append(j)
        if interaction:
            c = df["complexity_index"].astype(float)
            for col in j.columns:
                parts.append(pd.DataFrame({f"int_{col}": c * j[col]}))
    return pd.concat(parts, axis=1)


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, *, include_judge: bool, interaction: bool) -> dict:
    x_tr = make_design(train, include_judge=include_judge, interaction=interaction)
    x_te = make_design(test, include_judge=include_judge, interaction=interaction)
    x_te = x_te.reindex(columns=x_tr.columns, fill_value=0)

    y_tr = np.log1p(train["los_days"])
    reg = LinearRegression()
    reg.fit(x_tr, y_tr)

    pred_log = reg.predict(x_te)
    pred = np.expm1(np.clip(pred_log, 0, 10))  # cap ~22k days on original scale
    y_true = test["los_days"].values
    return {
        "R2": float(r2_score(y_true, pred)),
        "R2_log_scale": float(r2_score(np.log1p(y_true), pred_log)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(mean_squared_error(y_true, pred) ** 0.5),
        "n_features": int(x_tr.shape[1]),
    }


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "case_features.parquet")
    df = df[df["los_days"].notna()].copy()
    if "judge_concurrent_overlap" not in df.columns:
        raise SystemExit("Missing judge_concurrent_overlap; run add_judge_workload.py")

    top_suit = df["nature_suit"].value_counts().head(15).index
    df["nature_suit_top"] = df["nature_suit"].where(df["nature_suit"].isin(top_suit), "Other")

    judge_counts = df["District_Judge"].value_counts()
    top_judges = judge_counts[judge_counts >= 200].index
    df = df[df["District_Judge"].isin(top_judges)].copy()

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    m1 = fit_eval(train, test, include_judge=False, interaction=False)
    m2 = fit_eval(train, test, include_judge=True, interaction=False)
    m3 = fit_eval(train, test, include_judge=True, interaction=True)

    out = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "M1_complexity_suit": m1,
        "M2_add_judge": m2,
        "M3_complexity_x_judge": m3,
        "M2_R2_gain_over_M1": m2["R2"] - m1["R2"],
        "note": "Target=log1p(los_days). sklearn LinearRegression.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
