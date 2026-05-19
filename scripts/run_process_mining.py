#!/usr/bin/env python3
"""
Step 6: lightweight process mining + anomaly detection.

- Top process variants (activity sequences) on a case sample
- Bottleneck: median days motion -> next order
- Transition matrix (top edges)
- LOS anomalies: high residuals after complexity + judge controls
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
EVENT_LOG = ROOT / "Event Log.csv"
CASE_FEATURES = ROOT / "data" / "case_features.parquet"
OUT = ROOT / "docs" / "step6_process_mining.json"
FIG_DIR = ROOT / "reports" / "figures"


def parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def load_case_sample(n_cases: int, seed: int) -> set[str]:
    df = pd.read_parquet(CASE_FEATURES, columns=["ucid", "case_type", "case_status"])
    df = df[(df["case_type"] == "cv") & (df["case_status"] == "closed")]
    return set(df["ucid"].sample(min(n_cases, len(df)), random_state=seed))


def stream_traces(ucids: set[str], max_rows: int | None) -> dict[str, list[tuple[datetime, str]]]:
    traces: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    n = 0
    with EVENT_LOG.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            u = row["ucid"]
            if u not in ucids:
                continue
            if row.get("case_type") != "cv":
                continue
            d = parse_date(row.get("date_filed", ""))
            act = (row.get("Activity") or "").strip()
            if d and act:
                traces[u].append((d, act))
            if max_rows and n >= max_rows:
                break
    for u in traces:
        traces[u].sort(key=lambda x: x[0])
    return traces


def variant_key(events: list[str], max_len: int = 12) -> str:
    return " > ".join(events[:max_len]) + (" > ..." if len(events) > max_len else "")


def process_variants(traces: dict[str, list]) -> dict:
    variants: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    motion_to_order_days: list[float] = []

    for seq in traces.values():
        acts = [a for _, a in seq]
        variants[variant_key(acts)] += 1
        for i in range(len(acts) - 1):
            transitions[(acts[i], acts[i + 1])] += 1

        # bottleneck: motion -> next order
        for i, (d, a) in enumerate(seq):
            if a != "motion":
                continue
            for d2, a2 in seq[i + 1 :]:
                if a2 == "order":
                    motion_to_order_days.append((d2 - d).days)
                    break

    top_variants = variants.most_common(15)
    top_transitions = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in transitions.most_common(20)
    ]
    bottleneck = {
        "motion_to_order_n": len(motion_to_order_days),
        "motion_to_order_median_days": float(np.median(motion_to_order_days)) if motion_to_order_days else None,
        "motion_to_order_p90_days": float(np.percentile(motion_to_order_days, 90)) if motion_to_order_days else None,
    }
    return {
        "n_cases_with_traces": len(traces),
        "n_unique_variants": len(variants),
        "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
        "top_transitions": top_transitions,
        "bottleneck_motion_to_order": bottleneck,
    }


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Flag cases with largest LOS residual after complexity + judge."""
    top_suit = df["nature_suit"].value_counts().head(15).index
    df = df.copy()
    df["nature_suit_top"] = df["nature_suit"].where(df["nature_suit"].isin(top_suit), "Other")
    judge_counts = df["District_Judge"].value_counts()
    top_judges = judge_counts[judge_counts >= 200].index
    df = df[df["District_Judge"].isin(top_judges)].copy()

    x = pd.concat(
        [
            df[["complexity_index"]].astype(float),
            pd.get_dummies(df["nature_suit_top"], prefix="suit", dtype=float),
            pd.get_dummies(df["District_Judge"], prefix="judge", dtype=float),
        ],
        axis=1,
    )
    y = np.log1p(df["los_days"])
    reg = LinearRegression().fit(x, y)
    pred_log = reg.predict(x)
    df["residual_log"] = y - pred_log
    df["pred_los"] = np.expm1(np.clip(pred_log, 0, 10))
    df["residual_days"] = df["los_days"] - df["pred_los"]

    threshold = df["residual_log"].quantile(0.99)
    anomalies = df[df["residual_log"] >= threshold].copy()

    judge_anom = anomalies["District_Judge"].value_counts()
    judge_all = df["District_Judge"].value_counts()
    rate = (judge_anom / judge_all).dropna().sort_values(ascending=False)

    # save figure: anomaly rate by judge (top 15)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rates = rate.head(15)
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_rates.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Anomaly rate (99th pct residual)")
    ax.set_title("LOS anomalies by judge (after complexity+suit controls)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_anomaly_rate_by_judge.png", dpi=120)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x="complexity_index",
        y="residual_days",
        alpha=0.2,
        ax=ax2,
    )
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("LOS residual vs complexity")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "08_residual_vs_complexity.png", dpi=120)
    plt.close()

    return {
        "n_cases": int(len(df)),
        "n_anomalies_top1pct": int(len(anomalies)),
        "residual_threshold_log": float(threshold),
        "median_residual_days_anomalies": float(anomalies["residual_days"].median()),
        "top_judges_by_anomaly_rate": {k: float(v) for k, v in rate.head(10).items()},
        "example_anomalies": anomalies.nlargest(5, "residual_days")[
            ["ucid", "District_Judge", "los_days", "pred_los", "residual_days", "complexity_index"]
        ].to_dict(orient="records"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-cases", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None, help="Cap CSV rows scanned")
    args = p.parse_args()

    print(f"Sampling {args.sample_cases} cases ...")
    ucids = load_case_sample(args.sample_cases, args.seed)
    print(f"Streaming traces for {len(ucids)} cases ...")
    traces = stream_traces(ucids, args.max_rows)

    pm = process_variants(traces)
    pm["sample_cases_requested"] = args.sample_cases

    print("Detecting LOS anomalies on full case table ...")
    df = pd.read_parquet(CASE_FEATURES)
    df = df[df["los_days"].notna() & (df["case_type"] == "cv")]
    anom = detect_anomalies(df)

    out = {"process_mining": pm, "anomalies": anom}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"Saved -> {OUT}")
    print(f"Figures -> {FIG_DIR}/07_*, 08_*")


if __name__ == "__main__":
    main()
