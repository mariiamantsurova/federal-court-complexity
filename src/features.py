"""Shared feature definitions for case-level modeling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COMPLEXITY_CORE = [
    "n_events",
    "n_activity_types",
    "n_motions",
    "activity_entropy",
]

COMPLEXITY_NUMERIC = [
    *COMPLEXITY_CORE,
    "plaintiffs_count",
    "plaintiffs_counsels_count",
    "Defendants_count",
    "Defendants_counsels_count",
    "Defendants_pending_counts",
    "Defendants_terminated_counts",
    "related_case_count",
]

COMPLEXITY_CATEGORICAL = ["case_type", "city", "is_mdl", "District_Judge", "Magistrate_Judge"]

VALID_CASE_TYPES = ("cv", "cr")

TARGET = "los_days"
LOG_TARGET = "log_los_days"


def normalize_case_type(case_type: str | None) -> str | None:
    if case_type is None or case_type == "all":
        return None
    if case_type not in VALID_CASE_TYPES:
        raise ValueError(f"case_type must be one of all, {', '.join(VALID_CASE_TYPES)}; got {case_type!r}")
    return case_type


def case_type_suffix(case_type: str | None) -> str:
    ct = normalize_case_type(case_type)
    return f"_{ct}" if ct else ""


def filter_by_case_type(df: pd.DataFrame, case_type: str | None) -> pd.DataFrame:
    ct = normalize_case_type(case_type)
    if ct is None:
        return df
    if "case_type" not in df.columns:
        raise ValueError("case_type column missing; cannot filter by case type")
    return df.loc[df["case_type"] == ct].copy()


def filter_exclude_mdl(df: pd.DataFrame) -> pd.DataFrame:
    """Drop MDL cases (is_mdl == True)."""
    if "is_mdl" not in df.columns:
        return df
    return df.loc[df["is_mdl"] != True].copy()  # noqa: E712


def apply_data_filters(
    df: pd.DataFrame,
    *,
    case_type: str | None = None,
    exclude_mdl: bool = False,
) -> pd.DataFrame:
    work = filter_by_case_type(df, case_type)
    if exclude_mdl:
        work = filter_exclude_mdl(work)
    return work


def filter_suffix(case_type: str | None = None, exclude_mdl: bool = False) -> str:
    parts = [case_type_suffix(case_type)]
    if exclude_mdl:
        parts.append("_no_mdl")
    return "".join(p for p in parts if p)


def categorical_cols(case_type: str | None = None, *, exclude_mdl: bool = False) -> list[str]:
    cols = list(COMPLEXITY_CATEGORICAL)
    if normalize_case_type(case_type) is not None:
        cols = [c for c in cols if c != "case_type"]
    if exclude_mdl:
        cols = [c for c in cols if c != "is_mdl"]
    return cols


def tagged_path(base: Path, case_type: str | None = None, *, exclude_mdl: bool = False) -> Path:
    suffix = filter_suffix(case_type, exclude_mdl)
    if not suffix:
        return base
    return base.parent / f"{base.stem}{suffix}{base.suffix}"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log target and composite complexity index (z-scored core metrics)."""
    out = df.copy()
    if TARGET in out.columns:
        out[LOG_TARGET] = np.log1p(out[TARGET].clip(lower=0))
    present = [c for c in COMPLEXITY_CORE if c in out.columns]
    if present:
        z = out[present].apply(pd.to_numeric, errors="coerce")
        z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
        out["complexity_index"] = z.mean(axis=1)
    return out


def prepare_case_model_frame(
    df: pd.DataFrame,
    *,
    case_type: str | None = None,
    exclude_mdl: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Closed cases with valid LOS; returns X, y, numeric cols, categorical cols."""
    work = apply_data_filters(df, case_type=case_type, exclude_mdl=exclude_mdl)
    work = work.dropna(subset=[TARGET]).copy()
    work = work[work[TARGET] >= 0]
    work = add_derived_columns(work)

    sum_cols = sorted(c for c in work.columns if c.startswith("sum_attribute_"))
    numeric = [c for c in COMPLEXITY_NUMERIC if c in work.columns] + sum_cols
    categorical = [c for c in categorical_cols(case_type, exclude_mdl=exclude_mdl) if c in work.columns]

    X = work[numeric + categorical]
    y = work[TARGET]
    return X, y, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols if isinstance(cols, list) else [cols])
    return names
