#!/usr/bin/env python3
"""Random Forest for LOS prediction; save metrics and feature importance."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "step5_ml.json"
FIG = ROOT / "reports" / "figures" / "06_rf_feature_importance.png"

# Full set (includes volume/timing features — partly mechanical with LOS)
NUMERIC_FULL = [
    "complexity_index",
    "judge_concurrent_overlap",
    "n_events",
    "n_motions",
    "n_orders",
    "n_notices",
    "n_activity_types",
    "activity_entropy",
    "n_attribute_flags",
    "rework_ratio",
    "time_gaps_std",
    "party_load",
    "counsel_load",
    "pro_se_parties",
    "judge_caseload",
    "related_case_count",
]

# Restricted: structural complexity only (fairer for causal narrative)
NUMERIC_RESTRICTED = [
    "complexity_index",
    "judge_concurrent_overlap",
    "n_motions",
    "n_activity_types",
    "activity_entropy",
    "rework_ratio",
    "party_load",
    "counsel_load",
    "pro_se_parties",
    "judge_caseload",
    "related_case_count",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    top_suit = df["nature_suit"].value_counts().head(15).index
    df = df.copy()
    df["nature_suit_top"] = df["nature_suit"].where(df["nature_suit"].isin(top_suit), "Other")
    judge_counts = df["District_Judge"].value_counts()
    top_judges = judge_counts[judge_counts >= 200].index
    return df[df["District_Judge"].isin(top_judges)]


def make_X(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    nums = [c for c in numeric if c in df.columns]
    parts = [df[nums].astype(float)]
    parts.append(pd.get_dummies(df["nature_suit_top"], prefix="suit", dtype=float))
    parts.append(pd.get_dummies(df["District_Judge"], prefix="judge", dtype=float))
    return pd.concat(parts, axis=1)


def feature_group(name: str) -> str:
    if name in NUMERIC_FULL or name.startswith("z_"):
        return "complexity"
    if name.startswith("judge_"):
        return "judge"
    if name.startswith("suit_"):
        return "case_type"
    return "other"


def run_rf(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str]) -> tuple[dict, pd.Series]:
    x_tr = make_X(train, numeric)
    x_te = make_X(test, numeric).reindex(columns=x_tr.columns, fill_value=0)
    y_tr = np.log1p(train["los_days"])
    y_te_log = np.log1p(test["los_days"])
    y_te = test["los_days"].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_tr, y_tr)
    pred_log = rf.predict(x_te)
    pred = np.expm1(np.clip(pred_log, 0, 10))
    imp = pd.Series(rf.feature_importances_, index=x_tr.columns).sort_values(ascending=False)
    metrics = {
        "R2_log_scale": float(r2_score(y_te_log, pred_log)),
        "MAE_days": float(mean_absolute_error(y_te, pred)),
        "RMSE_days": float(mean_squared_error(y_te, pred) ** 0.5),
        "top_10_features": {k: float(v) for k, v in imp.head(10).items()},
        "importance_by_group": {k: float(v) for k, v in imp.groupby(imp.index.map(feature_group)).sum().items()},
    }
    return metrics, imp


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "case_features.parquet")
    df = df[df["los_days"].notna()]
    df = prepare(df)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    full_metrics, imp = run_rf(train, test, NUMERIC_FULL)
    restricted_metrics, _ = run_rf(train, test, NUMERIC_RESTRICTED)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Random Forest (full features) — top 15 importances")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    plt.close()

    out = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "RF_full": full_metrics,
        "RF_restricted_no_event_volume": restricted_metrics,
        "caveat": "n_events and time_gaps_std are mechanically tied to LOS; prefer RF_restricted for interpretation.",
        "compare_regression_M3_R2_log": 0.547,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Figure -> {FIG}")


if __name__ == "__main__":
    main()
