#!/usr/bin/env python3
"""
Build polished summary tables from docs/*_results.json for the report.

Usage:
  python scripts/build_results_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TABLES = DOCS / "tables"
CASE_TYPES = ("all", "cv", "cr")


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _ml_path(case_type: str, no_mdl: bool = False) -> Path:
    stem = "01_ml_results"
    if case_type != "all":
        stem += f"_{case_type}"
    if no_mdl:
        stem += "_no_mdl"
    return DOCS / f"{stem}.json"


def _xgb_path(case_type: str, no_mdl: bool = False) -> Path:
    stem = "02_xgb_shap_results"
    if case_type != "all":
        stem += f"_{case_type}"
    if no_mdl:
        stem += "_no_mdl"
    return DOCS / f"{stem}.json"


def _causal_path(case_type: str, no_mdl: bool = False) -> Path:
    stem = "03_causal_regression_results"
    if case_type != "all":
        stem += f"_{case_type}"
    if no_mdl:
        stem += "_no_mdl"
    return DOCS / f"{stem}.json"


def build_predictive_metrics_table() -> pd.DataFrame:
    rows: list[dict] = []
    for sample, no_mdl in (("all_cases", False), ("excl_mdl", True)):
        for ct in CASE_TYPES:
            ml = _load_json(_ml_path(ct, no_mdl))
            xgb = _load_json(_xgb_path(ct, no_mdl))
            if not ml and not xgb:
                continue
            base = ml or xgb
            for model_name, label in (
                ("decision_tree", "Decision Tree"),
                ("random_forest", "Random Forest"),
            ):
                if ml and model_name in ml.get("models", {}):
                    m = ml["models"][model_name]["metrics"]
                    rows.append({
                        "sample": sample,
                        "case_type": ct,
                        "model": label,
                        "n_cases": base.get("n_cases"),
                        "n_test": base.get("n_test"),
                        "mae_days": round(m["mae"], 1),
                        "rmse_days": round(m["rmse"], 1),
                        "r2_test": round(m["r2"], 4),
                    })
            if xgb:
                m = xgb["metrics"]
                rows.append({
                    "sample": sample,
                    "case_type": ct,
                    "model": "XGBoost",
                    "n_cases": xgb.get("n_cases"),
                    "n_test": xgb.get("n_test"),
                    "mae_days": round(m["mae"], 1),
                    "rmse_days": round(m["rmse"], 1),
                    "r2_test": round(m["r2"], 4),
                })
    return pd.DataFrame(rows)


def build_causal_r2_table() -> pd.DataFrame:
    rows: list[dict] = []
    for sample, no_mdl in (("all_cases", False), ("excl_mdl", True)):
        for ct in CASE_TYPES:
            data = _load_json(_causal_path(ct, no_mdl))
            if not data:
                continue
            for model_name, info in data.get("models", {}).items():
                rows.append({
                    "sample": sample,
                    "case_type": ct,
                    "model": model_name,
                    "n_cases": data.get("n_cases"),
                    "r2": round(info["r2"], 4),
                    "r2_adj": round(info["r2_adj"], 4),
                })
    return pd.DataFrame(rows)


def _coef_path(case_type: str, no_mdl: bool = False) -> Path:
    name = "03_causal_regression_coefs"
    if case_type != "all":
        name += f"_{case_type}"
    if no_mdl:
        name += "_no_mdl"
    return DOCS / f"{name}.csv"


def build_causal_m3_coefs_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sample, no_mdl in (("all_cases", False), ("excl_mdl", True)):
        for ct in CASE_TYPES:
            path = _coef_path(ct, no_mdl)
            if not path.is_file():
                continue
            m3 = pd.read_csv(path)
            m3 = m3[m3["model"] == "M3_with_judge_fe"].copy()
            m3["sample"] = sample
            m3["case_type"] = ct
            m3["term"] = m3["term"].str.replace("z_", "", regex=False)
            rows.append(m3[["sample", "case_type", "term", "coef", "p_value"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["coef"] = out["coef"].round(4)
    out["p_value"] = out["p_value"].apply(lambda p: f"{p:.2e}")
    return out.rename(columns={
        "term": "complexity_feature",
        "coef": "M3_coef_log_los",
        "p_value": "M3_p_value",
    })


def build_mdl_counts_table() -> pd.DataFrame:
    path = ROOT / "data" / "aggregations" / "by_case.parquet"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["case_type", "is_mdl", "los_days"])
    rows = []
    for ct in ("all", "cv", "cr"):
        sub = df if ct == "all" else df[df["case_type"] == ct]
        mdl = sub[sub["is_mdl"] == True]  # noqa: E712
        rows.append({
            "case_type": ct,
            "n_total": len(sub),
            "n_mdl": len(mdl),
            "pct_mdl": round(100 * len(mdl) / len(sub), 2) if len(sub) else 0,
            "median_los_all": round(sub["los_days"].median(), 0),
            "median_los_mdl": round(mdl["los_days"].median(), 0) if len(mdl) else None,
            "median_los_non_mdl": round(sub[sub["is_mdl"] != True]["los_days"].median(), 0),  # noqa: E712
        })
    return pd.DataFrame(rows)


def build_mdl_sensitivity_delta() -> pd.DataFrame:
    pred = build_predictive_metrics_table()
    if pred.empty or "excl_mdl" not in pred["sample"].values:
        return pd.DataFrame()
    base = pred[pred["sample"] == "all_cases"].set_index(["case_type", "model"])
    excl = pred[pred["sample"] == "excl_mdl"].set_index(["case_type", "model"])
    common = base.index.intersection(excl.index)
    rows = []
    for idx in common:
        rows.append({
            "case_type": idx[0],
            "model": idx[1],
            "r2_all_cases": base.loc[idx, "r2_test"],
            "r2_excl_mdl": excl.loc[idx, "r2_test"],
            "delta_r2": round(excl.loc[idx, "r2_test"] - base.loc[idx, "r2_test"], 4),
            "mae_all_cases": base.loc[idx, "mae_days"],
            "mae_excl_mdl": excl.loc[idx, "mae_days"],
            "delta_mae_days": round(excl.loc[idx, "mae_days"] - base.loc[idx, "mae_days"], 1),
        })
    return pd.DataFrame(rows)


def build_all_tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    tables = {
        "T1_predictive_metrics.csv": build_predictive_metrics_table(),
        "T2_causal_regression_r2.csv": build_causal_r2_table(),
        "T3_causal_m3_complexity_coefs.csv": build_causal_m3_coefs_table(),
        "T4_mdl_case_counts.csv": build_mdl_counts_table(),
        "T5_mdl_sensitivity_delta.csv": build_mdl_sensitivity_delta(),
    }
    for name, df in tables.items():
        out = TABLES / name
        if df.empty:
            print(f"skip (empty): {name}")
            continue
        df.to_csv(out, index=False)
        print(f"saved -> {out} ({len(df)} rows)")


def main() -> None:
    build_all_tables()


if __name__ == "__main__":
    main()
