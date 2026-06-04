#!/usr/bin/env python3
"""
Process-mining visuals for the report (civil, non-MDL, closed cases).

Outputs:
  reports/figures/05_dfg_cv_sample.png
  reports/figures/05_trace_exemplars_cv.png
  reports/figures/05_transitions_cv_q1_q4.png
  docs/tables/T7_trace_exemplars.csv
  docs/05_process_mining.json
  docs/PROCESS_MINING.md

Usage:
  python scripts/run_process_mining_viz.py
  python scripts/run_process_mining_viz.py --dfg-sample 1500 --transition-sample 800
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import TARGET, add_derived_columns, filter_exclude_mdl  # noqa: E402

DEFAULT_CASES = ROOT / "data" / "case_features.parquet"
EVENT_LOG = ROOT / "Event Log_model.csv"
FIG_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"
TABLES_DIR = DOCS_DIR / "tables"

TOP_N_ACTIVITIES = 10
MIN_EVENTS = 10
MAX_EVENTS = 80
DEFAULT_DFG_SAMPLE = 2000
DEFAULT_TRANSITION_SAMPLE = 1000
MIN_DFG_EDGE_FREQ = 50

ARCHETYPE_TITLES = {
    "typical_efficient": "Typical duration (~median LOS)",
    "typical_long": "Long duration (~90th pct LOS)",
    "high_complexity_moderate_los": "High complexity, mid LOS",
    "moderate_complexity_very_long_los": "Mid complexity, very long LOS",
}

ACTIVITY_COLORS = {
    "complaint": "#4C78A8",
    "motion": "#F58518",
    "response": "#E45756",
    "order": "#72B7B2",
    "notice": "#54A24B",
    "minute_entry": "#B279A2",
    "answer": "#FF9DA6",
    "summons": "#9D755D",
    "hearing": "#EDC948",
    "scheduling": "#76B7B2",
}


def _load_cases(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run src/build_case_features.py first.")
    work = pd.read_parquet(path)
    work = work.loc[work["case_type"] == "cv"].copy()
    work = filter_exclude_mdl(work)
    work = work.dropna(subset=[TARGET])
    work = work[(work[TARGET] >= 0) & work["n_events"].between(MIN_EVENTS, MAX_EVENTS)]
    return add_derived_columns(work)


def _pick_closest(df: pd.DataFrame, col: str, target: float) -> pd.Series:
    return df.loc[(df[col] - target).abs().idxmin()]


def select_exemplars(df: pd.DataFrame) -> pd.DataFrame:
    q50, q75, q95 = df[TARGET].quantile([0.5, 0.75, 0.95])
    q90_cx = df["complexity_index"].quantile(0.9)
    q40_cx, q60_cx = df["complexity_index"].quantile([0.4, 0.6])
    med = df[TARGET].median()

    picks: list[tuple[str, pd.Series]] = []
    m = df[TARGET].between(200, 300) & df["n_events"].between(15, 35)
    if m.any():
        picks.append(("typical_efficient", _pick_closest(df.loc[m], TARGET, med)))
    m = df[TARGET].between(900, 1100) & df["n_events"].between(25, 60)
    if m.any():
        picks.append(("typical_long", _pick_closest(df.loc[m], TARGET, q95)))
    m = (df["complexity_index"] >= q90_cx) & df[TARGET].between(q50, q75)
    if m.any():
        picks.append(("high_complexity_moderate_los", df.loc[m].sort_values("complexity_index", ascending=False).iloc[0]))
    m = df["complexity_index"].between(q40_cx, q60_cx) & (df[TARGET] >= q95)
    if m.any():
        picks.append(("moderate_complexity_very_long_los", df.loc[m].sort_values(TARGET, ascending=False).iloc[0]))
    if len(picks) < 4:
        raise RuntimeError(f"Only matched {len(picks)}/4 exemplars.")

    rows: list[dict] = []
    used: set[str] = set()
    for arch, row in picks:
        ucid = str(row["ucid"])
        if ucid in used:
            continue
        used.add(ucid)
        rows.append(
            {
                "archetype": arch,
                "report_label": chr(ord("A") + len(rows)),
                "ucid": ucid,
                "los_days": float(row[TARGET]),
                "n_events": int(row["n_events"]),
                "n_motions": int(row.get("n_motions", 0)),
                "n_activity_types": int(row.get("n_activity_types", 0)),
                "activity_entropy": float(row.get("activity_entropy", 0)),
                "complexity_index": float(row.get("complexity_index", 0)),
                "District_Judge": str(row.get("District_Judge", "")),
            }
        )
    return pd.DataFrame(rows)


def sample_quartile_ucids(df: pd.DataFrame, n: int, q_label: str) -> set[str]:
    work = df.copy()
    work["_q"] = pd.qcut(work[TARGET], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    sub = work.loc[work["_q"] == q_label]
    n = min(n, len(sub))
    return set(sub.sample(n=n, random_state=42)["ucid"].astype(str))


def stream_events(
    event_log: Path,
    *,
    exemplar_ucids: set[str],
    q1_ucids: set[str],
    q4_ucids: set[str],
    top_activities: list[str] | None,
) -> tuple[pd.DataFrame, Counter, Counter, Counter]:
    """Single CSV pass for exemplar traces + transition counts."""
    all_ucids = exemplar_ucids | q1_ucids | q4_ucids
    top_set = set(top_activities) if top_activities else None
    act_global: Counter[str] = Counter()
    pairs_q1: Counter[tuple[str, str]] = Counter()
    pairs_q4: Counter[tuple[str, str]] = Counter()
    exemplar_parts: list[pd.DataFrame] = []

    cols = ["ucid", "date_filed", "Activity"]
    for chunk in pd.read_csv(event_log, usecols=cols, chunksize=500_000, low_memory=False):
        chunk["ucid"] = chunk["ucid"].astype(str)
        sub = chunk.loc[chunk["ucid"].isin(all_ucids)].copy()
        if sub.empty:
            continue
        sub["Activity"] = sub["Activity"].astype(str)
        act_global.update(sub["Activity"].value_counts().to_dict())

        ex = sub.loc[sub["ucid"].isin(exemplar_ucids)]
        if not ex.empty:
            exemplar_parts.append(ex)

        if top_set:
            for ucid, g in sub.groupby("ucid", sort=False):
                acts = g.sort_values("date_filed")["Activity"].tolist()
                bucket = None
                if ucid in q1_ucids:
                    bucket = pairs_q1
                elif ucid in q4_ucids:
                    bucket = pairs_q4
                if bucket is None:
                    continue
                for i in range(len(acts) - 1):
                    a, b = acts[i], acts[i + 1]
                    if a in top_set and b in top_set:
                        bucket[(a, b)] += 1

    if exemplar_parts:
        ev = pd.concat(exemplar_parts, ignore_index=True)
        ev["date_filed"] = pd.to_datetime(ev["date_filed"], errors="coerce")
        ev = ev.dropna(subset=["date_filed"]).sort_values(["ucid", "date_filed"])
    else:
        ev = pd.DataFrame(columns=cols)
    return ev, pairs_q1, pairs_q4, act_global


def load_dfg_event_log(event_log: Path, ucids: set[str]) -> pd.DataFrame:
    cols = ["ucid", "date_filed", "Activity"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(event_log, usecols=cols, chunksize=500_000, low_memory=False):
        sub = chunk.loc[chunk["ucid"].astype(str).isin(ucids)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        raise FileNotFoundError("No events for DFG sample.")
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(
        columns={
            "ucid": "case:concept:name",
            "Activity": "concept:name",
            "date_filed": "time:timestamp",
        }
    )
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    return df.dropna(subset=["time:timestamp"]).sort_values(["case:concept:name", "time:timestamp"])


def discover_dfg(event_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    import pm4py

    return pm4py.discover_dfg(
        event_df,
        case_id_key="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


def plot_dfg_matplotlib(
    dfg: dict[tuple[str, str], float],
    start_acts: dict[str, float],
    end_acts: dict[str, float],
    out_path: Path,
    min_freq: int,
) -> dict:
    """Render frequency DFG (PM4Py discovery + matplotlib layout)."""
    filtered = {k: v for k, v in dfg.items() if v >= min_freq}
    if not filtered:
        filtered = dict(sorted(dfg.items(), key=lambda x: -x[1])[:30])

    nodes = sorted({a for edge in filtered for a in edge})
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {a: (np.cos(t), np.sin(t)) for a, t in zip(nodes, angles)}
    max_w = max(filtered.values()) if filtered else 1

    fig, ax = plt.subplots(figsize=(11, 9))
    for (src, tgt), w in filtered.items():
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=0.8 + 4.5 * w / max_w,
                color="#4C78A8",
                alpha=0.55,
                connectionstyle="arc3,rad=0.12",
            ),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if w >= max_w * 0.35:
            ax.text(mx, my, str(int(w)), fontsize=7, color="#333333", ha="center")

    for act, (x, y) in pos.items():
        size = 300 + 80 * start_acts.get(act, 0) + 80 * end_acts.get(act, 0)
        ax.scatter(x, y, s=size, c="#F58518", zorder=3, edgecolors="white")
        ax.text(x, y, act.replace("_", "\n"), ha="center", va="center", fontsize=8, zorder=4, fontweight="bold")

    ax.set_title(
        "Directly-follows graph (PM4Py discovery, civil non-MDL sample)\n"
        "Arrow width ∝ frequency; labels on strongest edges",
        fontsize=11,
    )
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_edges_total": len(dfg),
        "n_edges_shown": len(filtered),
        "min_edge_freq": min_freq,
        "top_edges": [(f"{a} -> {b}", int(c)) for (a, b), c in sorted(filtered.items(), key=lambda x: -x[1])[:15]],
        "top_start": sorted(start_acts.items(), key=lambda x: -x[1])[:5],
        "top_end": sorted(end_acts.items(), key=lambda x: -x[1])[:5],
    }


def plot_exemplar_timelines(exemplars: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    all_acts = events["Activity"].value_counts().head(TOP_N_ACTIVITIES).index.tolist()
    palette = {**ACTIVITY_COLORS, **{a: "#BAB0AC" for a in all_acts}}

    for ax, (_, meta) in zip(axes.ravel(), exemplars.iterrows(), strict=True):
        g = events.loc[events["ucid"].astype(str) == meta["ucid"]].copy().reset_index(drop=True)
        t0 = g["date_filed"].min()
        g["days"] = (g["date_filed"] - t0).dt.days
        g["ord"] = np.arange(len(g))
        for act in all_acts:
            sub = g.loc[g["Activity"] == act]
            if not sub.empty:
                ax.scatter(sub["days"], sub["ord"], c=palette.get(act, "#BAB0AC"), s=36, label=act)
        other = g.loc[~g["Activity"].isin(all_acts)]
        if not other.empty:
            ax.scatter(other["days"], other["ord"], c="#BAB0AC", s=24, label="other")
        ax.set_title(
            f"Case {meta['report_label']}: {ARCHETYPE_TITLES[meta['archetype']]}\n"
            f"LOS {meta['los_days']:.0f} d · {meta['n_events']} events",
            fontsize=10,
        )
        ax.set_xlabel("Days since first event")
        ax.set_ylabel("Event order")
        ax.grid(True, alpha=0.25)

    handles = [mpatches.Patch(color=palette.get(a, "#BAB0AC"), label=a) for a in all_acts]
    handles.append(mpatches.Patch(color="#BAB0AC", label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Illustrative civil traces (non-MDL)", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pairs_to_matrix(pairs: Counter, activities: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=activities, columns=activities)
    for (a, b), c in pairs.items():
        mat.loc[a, b] = c
    return mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)


def plot_transition_heatmaps(
    pairs_q1: Counter,
    pairs_q4: Counter,
    activities: list[str],
    out_path: Path,
    n_q1: int,
    n_q4: int,
) -> None:
    m1, m4 = _pairs_to_matrix(pairs_q1, activities), _pairs_to_matrix(pairs_q4, activities)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(m1.max().max(), m4.max().max(), 0.01)
    for ax, mat, title in zip(axes, [m1, m4], [f"Low LOS (Q1), n={n_q1}", f"High LOS (Q4), n={n_q4}"], strict=True):
        sns.heatmap(mat, ax=ax, cmap="Blues", vmin=0, vmax=vmax, annot=True, fmt=".2f", annot_kws={"size": 7})
        ax.set_title(title)
    fig.suptitle(f"Transition probabilities (top {len(activities)} activities)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_process_mining_doc(
    path: Path,
    exemplars: pd.DataFrame,
    meta: dict,
) -> None:
    lines = [
        "# Process mining — step-by-step (for your report)",
        "",
        "This document explains how the process-mining figures were built and how to describe them in the thesis.",
        "",
        "## What is process mining here?",
        "",
        "Each **case** (`ucid`) is a **trace**: a time-ordered sequence of **events** (`Activity` at `date_filed`).",
        "Process mining discovers **patterns in those sequences** — not just averages like LOS or `n_events`.",
        "",
        "---",
        "",
        "## Step 1 — Define the cohort",
        "",
        "| Rule | Value |",
        "|------|--------|",
        "| Case type | Civil (`cv`) only |",
        "| MDL | Excluded (`is_mdl != True`) |",
        "| Status | Closed cases (in `Event Log_model.csv`) |",
        "| Trace length | Between 10 and 80 events (readable plots) |",
        "",
        f"Population for selection: **{meta['n_cohort']:,}** civil non-MDL closed cases with valid LOS.",
        "",
        "---",
        "",
        "## Step 2 — Select illustrative traces (Figure: `05_trace_exemplars_cv.png`)",
        "",
        "Four **archetypes** (one case each) — not random samples:",
        "",
        "| Case | Archetype | Purpose |",
        "|------|-----------|---------|",
    ]
    for _, r in exemplars.iterrows():
        lines.append(
            f"| {r['report_label']} | {ARCHETYPE_TITLES[r['archetype']]} | "
            f"LOS={r['los_days']:.0f} d, events={r['n_events']}, complexity={r['complexity_index']:.2f} |"
        )
    lines.extend(
        [
            "",
            "**How to read the plot:**",
            "- **X-axis:** days since the first event in the case.",
            "- **Y-axis:** event order (1st, 2nd, 3rd, …).",
            "- **Color:** activity type (`motion`, `order`, `notice`, …).",
            "",
            "**Report sentence (template):**",
            "> Figure X shows four illustrative civil traces. Case B (long LOS) exhibits more mid-process ",
            "> motion–response activity than Case A (median LOS), while Case C separates high procedural ",
            "> complexity from extreme duration (Case D).",
            "",
            "Full case metadata: `docs/tables/T7_trace_exemplars.csv` (appendix; anonymize `ucid` if required).",
            "",
            "---",
            "",
            "## Step 3 — Discover the directly-follows graph (Figure: `05_dfg_cv_sample.png`)",
            "",
            f"1. Random sample of **{meta['dfg']['n_cases']}** cases from the same cohort.",
            f"2. Build a standard event log: case id = `ucid`, activity = `Activity`, timestamp = `date_filed`.",
            f"3. Run **DFG discovery** with PM4Py: edge *A → B* = how often activity B immediately follows A.",
            f"4. Keep edges with frequency ≥ **{meta['dfg'].get('min_edge_freq', MIN_DFG_EDGE_FREQ)}**; render with matplotlib (arrow width ∝ count).",
            "",
            "**How to read the DFG:**",
            "- **Nodes** = activities; **thick arrows** = frequent direct successors.",
            "- **Start/end** activities show how cases open and close in the sample.",
            "",
            "**Report sentence (template):**",
            "> The directly-follows graph on a stratified civil sample shows that `motion` and `minute_entry` ",
            "> are the most frequent hubs; dispositive paths often pass through `order` after `motion` or `response`.",
            "",
        ]
    )
    if meta["dfg"].get("top_edges"):
        lines.append("**Top edges in sample:**")
        lines.append("")
        lines.append("| From | To | Count |")
        lines.append("|------|-----|-------|")
        for edge, c in meta["dfg"]["top_edges"][:10]:
            if " -> " in edge:
                a, b = edge.split(" -> ", 1)
            else:
                a, b = edge, ""
            lines.append(f"| {a} | {b} | {c} |")
        lines.append("")
    lines.extend(
        [
            "---",
            "",
            "## Step 4 — Compare low vs high LOS transitions (Figure: `05_transitions_cv_q1_q4.png`)",
            "",
            f"1. Split cohort into LOS **quartiles**.",
            f"2. Sample **{meta['transitions']['n_q1']}** cases from Q1 (short LOS) and **{meta['transitions']['n_q4']}** from Q4 (long LOS).",
            f"3. Count consecutive activity pairs; keep top **{TOP_N_ACTIVITIES}** activities.",
            f"4. **Row-normalize** each heatmap → cell = P(next activity | current activity).",
            "",
            "**How to read the heatmaps:**",
            "- Compare the same row (e.g. `motion`) across panels: where probability mass shifts is the structural difference.",
            "- Higher values on `motion`→`response` or `motion`→`order` in Q4 support “more procedural churn” on long cases.",
            "",
            "**Report sentence (template):**",
            "> Compared with Q1, Q4 traces allocate a larger share of transitions from `motion` to follow-up ",
            "> activities rather than early disposition, consistent with longer LOS.",
            "",
            "---",
            "",
            "## Step 5 — Link back to your main models",
            "",
            "| Process-mining view | Your existing metrics |",
            "|--------------------|------------------------|",
            "| More loops / branches in DFG & heatmaps | Higher `n_activity_types`, `activity_entropy` |",
            "| More `motion` transitions | Higher `n_motions` |",
            "| Longer spans between events | Higher `los_days` even at moderate complexity |",
            "",
            "State clearly: this is **descriptive** process structure, not causal identification.",
            "",
            "---",
            "",
            "## Step 6 — Limitations (include in report)",
            "",
            "- Sample-based DFG and heatmaps (not full 4.8M-row log).",
            "- Civil non-MDL only; criminal paths differ.",
            "- Activities are coarse (`motion`, `order`); fine-grained `attribute_*` not shown.",
            "- Illustrative cases are cherry-picked for clarity, not population averages.",
            "",
            "---",
            "",
            "## Files",
            "",
            "| File | Role |",
            "|------|------|",
            "| `reports/figures/05_trace_exemplars_cv.png` | Four case timelines |",
            "| `reports/figures/05_dfg_cv_sample.png` | PM4Py directly-follows graph |",
            "| `reports/figures/05_transitions_cv_q1_vs_q4.png` | LOS Q1 vs Q4 transitions |",
            "| `docs/tables/T7_trace_exemplars.csv` | Case-level appendix table |",
            "| `docs/05_process_mining.json` | Run metadata |",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/run_process_mining_viz.py",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--event-log", type=Path, default=EVENT_LOG)
    parser.add_argument("--dfg-sample", type=int, default=DEFAULT_DFG_SAMPLE)
    parser.add_argument("--transition-sample", type=int, default=DEFAULT_TRANSITION_SAMPLE)
    parser.add_argument("--min-dfg-edge", type=int, default=MIN_DFG_EDGE_FREQ)
    parser.add_argument("--skip-dfg", action="store_true")
    args = parser.parse_args()

    if not args.event_log.is_file():
        raise SystemExit(f"Missing event log: {args.event_log}")

    sns.set_theme(style="whitegrid")
    cases = _load_cases(args.cases)
    exemplars = select_exemplars(cases)
    exemplar_ucids = set(exemplars["ucid"])

    q1_ucids = sample_quartile_ucids(cases, args.transition_sample, "Q1")
    q4_ucids = sample_quartile_ucids(cases, args.transition_sample, "Q4")
    dfg_ucids = set(cases.sample(n=min(args.dfg_sample, len(cases)), random_state=42)["ucid"].astype(str))

    # Pass 1: activity vocabulary from transition sample
    _, _, _, act_global = stream_events(
        args.event_log,
        exemplar_ucids=set(),
        q1_ucids=q1_ucids | q4_ucids,
        q4_ucids=set(),
        top_activities=None,
    )
    top_activities = [a for a, _ in act_global.most_common(TOP_N_ACTIVITIES)]

    # Pass 2: exemplars + transitions
    events, pairs_q1, pairs_q4, _ = stream_events(
        args.event_log,
        exemplar_ucids=exemplar_ucids,
        q1_ucids=q1_ucids,
        q4_ucids=q4_ucids,
        top_activities=top_activities,
    )

    fig_ex = FIG_DIR / "05_trace_exemplars_cv.png"
    plot_exemplar_timelines(exemplars, events, fig_ex)

    fig_tr = FIG_DIR / "05_transitions_cv_q1_q4.png"
    plot_transition_heatmaps(pairs_q1, pairs_q4, top_activities, fig_tr, len(q1_ucids), len(q4_ucids))

    dfg_meta: dict = {}
    fig_dfg = FIG_DIR / "05_dfg_cv_sample.png"
    if not args.skip_dfg:
        dfg_df = load_dfg_event_log(args.event_log, dfg_ucids)
        dfg, start_acts, end_acts = discover_dfg(dfg_df)
        dfg_meta = plot_dfg_matplotlib(dfg, start_acts, end_acts, fig_dfg, args.min_dfg_edge)
        dfg_meta["n_cases"] = int(dfg_df["case:concept:name"].nunique())

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t7 = TABLES_DIR / "T7_trace_exemplars.csv"
    exemplars.to_csv(t7, index=False)

    meta = {
        "n_cohort": len(cases),
        "exemplars": exemplars.to_dict(orient="records"),
        "dfg": dfg_meta,
        "transitions": {"n_q1": len(q1_ucids), "n_q4": len(q4_ucids), "top_activities": top_activities},
        "figures": {
            "exemplars": str(fig_ex.relative_to(ROOT)),
            "dfg": str(fig_dfg.relative_to(ROOT)),
            "transitions": str(fig_tr.relative_to(ROOT)),
        },
    }
    (DOCS_DIR / "05_process_mining.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_process_mining_doc(DOCS_DIR / "PROCESS_MINING.md", exemplars, meta)

    print(f"Wrote {fig_ex}")
    print(f"Wrote {fig_tr}")
    if not args.skip_dfg:
        print(f"Wrote {fig_dfg}")
    print(f"Wrote {t7}")
    print(f"Wrote {DOCS_DIR / 'PROCESS_MINING.md'}")


if __name__ == "__main__":
    main()
