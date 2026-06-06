#!/usr/bin/env python3
"""
Full pipeline orchestrator — federal court complexity project.

Steps:
  1  build_features   scripts/build_features.py          (Event Log.csv → Event Log_model.csv)
  2  build_cases      src/build_case_features.py         (Event Log_model.csv → data/case_features.parquet)
  3  build_agg        src/build_aggregations.py           (case_features.parquet → data/aggregations/)
  4  rf_shap          scripts/run_rf_shap.py              (Random Forest + SHAP)
  5  xgb_shap         scripts/run_xgb_shap.py             (XGBoost + SHAP)
  6  neural_net       scripts/run_neural_net.py            (Neural Network, learned embeddings)
  6b neural_net_hf    scripts/run_neural_net.py --use-hf-embeddings   (HuggingFace judge embeddings)

Skip flags let you resume from any point when upstream data already exists.

Usage examples:
  # Full run from scratch (needs Event Log.csv)
  .venv/bin/python3 scripts/run_pipeline.py

  # Skip Steps 1-2 (Event Log_model.csv and case_features.parquet already exist)
  .venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case

  # Skip Steps 1-3 (aggregations already exist), run models for civil cases only
  .venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case --skip-agg --case-type cv

  # Full run with civil + criminal split, exclude MDL, HF embeddings in NN
  .venv/bin/python3 scripts/run_pipeline.py --skip-clean --skip-case --by-case-type --exclude-mdl --use-hf-embeddings

  # Dev run: sample 500k events, then full models
  .venv/bin/python3 scripts/run_pipeline.py --sample 500000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python3"


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(msg: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _run(label: str, cmd: list[str | Path], *, check: bool = True) -> bool:
    """Run a subprocess, stream output, return True on success."""
    _header(label)
    print(f"$ {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status}  ({elapsed:.1f}s)")
    if not ok and check:
        sys.exit(f"\nPipeline aborted at step: {label}")
    return ok


def _check_file(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(
            f"\nPrerequisite missing: {path.relative_to(ROOT)}\n"
            f"  → needed for {label}\n"
            f"  → run without --skip-{label.split()[0].lower()} to build it first"
        )


# ── steps ─────────────────────────────────────────────────────────────────────

def step_build_features(args: argparse.Namespace) -> None:
    _check_file(ROOT / "Event Log.csv", "build_features")
    cmd = [PYTHON, "scripts/build_features.py"]
    if args.sample:
        cmd += ["--sample-rows", str(args.sample)]
    _run("Step 1 — build_features (Event Log.csv → Event Log_model.csv)", cmd)


def step_build_cases(args: argparse.Namespace) -> None:
    _check_file(ROOT / "Event Log_model.csv", "build_cases")
    cmd = [PYTHON, "src/build_case_features.py"]
    if args.sample:
        cmd += ["--sample-rows", str(args.sample)]
    _run("Step 2 — build_case_features (Event Log_model.csv → case_features.parquet)", cmd)


def step_build_agg(args: argparse.Namespace) -> None:
    _check_file(ROOT / "data" / "case_features.parquet", "build_agg")
    _run(
        "Step 3 — build_aggregations (case_features.parquet → data/aggregations/)",
        [PYTHON, "src/build_aggregations.py"],
    )


def _model_suffix(case_type: str | None, exclude_mdl: bool) -> str:
    parts = []
    if case_type:
        parts.append(case_type)
    if exclude_mdl:
        parts.append("no_mdl")
    return ("_" + "_".join(parts)) if parts else ""


def step_rf(args: argparse.Namespace, case_type: str | None) -> None:
    suffix = _model_suffix(case_type, args.exclude_mdl)
    cmd = [PYTHON, "scripts/run_rf_shap.py"]
    if case_type:
        cmd += ["--case-type", case_type]
    if args.exclude_mdl:
        cmd.append("--exclude-mdl")
    _run(f"Step 4 — RF + SHAP  [{case_type or 'all'}{' excl-MDL' if args.exclude_mdl else ''}]", cmd)


def step_xgb(args: argparse.Namespace, case_type: str | None) -> None:
    cmd = [PYTHON, "scripts/run_xgb_shap.py"]
    if case_type:
        cmd += ["--case-type", case_type]
    if args.exclude_mdl:
        cmd.append("--exclude-mdl")
    _run(f"Step 5 — XGBoost + SHAP  [{case_type or 'all'}{' excl-MDL' if args.exclude_mdl else ''}]", cmd)


def step_nn(args: argparse.Namespace, case_type: str | None, use_hf: bool) -> None:
    cmd = [PYTHON, "scripts/run_neural_net.py"]
    if case_type:
        cmd += ["--case-type", case_type]
    if args.exclude_mdl:
        cmd.append("--exclude-mdl")
    if use_hf:
        cmd.append("--use-hf-embeddings")
    if args.nn_epochs:
        cmd += ["--epochs", str(args.nn_epochs)]
    if args.nn_batch_size:
        cmd += ["--batch-size", str(args.nn_batch_size)]
    mode = "HF-embeddings" if use_hf else "learned-embeddings"
    _run(
        f"Step 6 — Neural Net [{case_type or 'all'} | {mode}"
        + (f" | excl-MDL" if args.exclude_mdl else "") + "]",
        cmd,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Skip flags
    parser.add_argument("--skip-clean",  action="store_true",
                        help="Skip Step 1 — reuse existing Event Log_model.csv")
    parser.add_argument("--skip-case",   action="store_true",
                        help="Skip Step 2 — reuse existing data/case_features.parquet")
    parser.add_argument("--skip-agg",    action="store_true",
                        help="Skip Step 3 — reuse existing data/aggregations/")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip Steps 4-6 (data prep only)")

    # Data options
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Pass --sample-rows N to build_features and build_case_features (dev mode)")

    # Model options
    parser.add_argument("--by-case-type", action="store_true",
                        help="Also run models separately for civil (cv) and criminal (cr)")
    parser.add_argument("--case-type", choices=["cv", "cr"], default=None,
                        help="Run models for one case type only (pooled is always run unless --case-type is set)")
    parser.add_argument("--exclude-mdl", action="store_true",
                        help="Pass --exclude-mdl to all model scripts")
    parser.add_argument("--use-hf-embeddings", action="store_true",
                        help="Also run Neural Net with HuggingFace sentence-transformer judge embeddings")
    parser.add_argument("--skip-nn",     action="store_true",
                        help="Skip Neural Net step (faster when only tree models needed)")

    # NN hyperparams
    parser.add_argument("--nn-epochs",      type=int, default=None)
    parser.add_argument("--nn-batch-size",  type=int, default=None)

    args = parser.parse_args()

    pipeline_start = time.time()
    print(f"\nFederal Court Complexity Pipeline")
    print(f"Root:   {ROOT}")
    print(f"Python: {PYTHON}")

    # Validate skip combinations
    if args.skip_case and not args.skip_clean:
        print("  (--skip-case implies also skipping clean; setting --skip-clean)")
        args.skip_clean = True
    if args.skip_agg and not args.skip_case:
        print("  (--skip-agg implies also skipping case; setting --skip-clean --skip-case)")
        args.skip_clean = True
        args.skip_case = True

    # Decide which case types to run models for
    if args.case_type:
        model_case_types = [args.case_type]
    elif args.by_case_type:
        model_case_types = [None, "cv", "cr"]
    else:
        model_case_types = [None]

    # ── Steps 1-3: data pipeline ──────────────────────────────────────────────
    if not args.skip_clean:
        step_build_features(args)
    else:
        _check_file(ROOT / "Event Log_model.csv", "build_cases (skipped clean)")
        print("\n[Step 1] Skipped — using existing Event Log_model.csv")

    if not args.skip_case:
        step_build_cases(args)
    else:
        _check_file(ROOT / "data" / "case_features.parquet", "build_agg (skipped case)")
        print("[Step 2] Skipped — using existing data/case_features.parquet")

    if not args.skip_agg:
        step_build_agg(args)
    else:
        _check_file(ROOT / "data" / "aggregations" / "by_case.parquet", "models (skipped agg)")
        print("[Step 3] Skipped — using existing data/aggregations/")

    if args.skip_models:
        print("\n[Steps 4-6] Skipped (--skip-models)")
        total = time.time() - pipeline_start
        print(f"\nPipeline complete (data only) in {total:.1f}s")
        return

    # ── Steps 4-6: models ─────────────────────────────────────────────────────
    for ct in model_case_types:
        step_rf(args, ct)
        step_xgb(args, ct)

        if not args.skip_nn:
            # Custom learned embeddings (always)
            step_nn(args, ct, use_hf=False)
            # HuggingFace variant (optional)
            if args.use_hf_embeddings:
                step_nn(args, ct, use_hf=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    _header("Pipeline complete")
    print(f"Total time: {total:.1f}s  ({total/60:.1f} min)")
    print()
    print("Outputs:")
    for p in sorted((ROOT / "docs").glob("0*.json")):
        print(f"  {p.relative_to(ROOT)}")
    for p in sorted((ROOT / "reports" / "figures").glob("0*.png")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
