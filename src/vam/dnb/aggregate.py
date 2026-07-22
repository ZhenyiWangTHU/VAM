"""DNB score aggregation."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def aggregate_idnb(scores: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Average each sample's top-k module scores."""

    required = {"sample_id", "dnb"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Scores must contain columns {sorted(required)}")
    group_columns = ["sample_id"] + (["stage"] if "stage" in scores.columns else [])
    return (
        scores.sort_values("dnb", ascending=False)
        .groupby(group_columns, as_index=False)
        .head(top_k)
        .groupby(group_columns, as_index=False)["dnb"]
        .mean()
        .rename(columns={"dnb": "idnb"})
    )


def aggregate_stages(
    idnb: pd.DataFrame,
    method: Literal["mean", "max", "median"] = "max",
) -> pd.DataFrame:
    """Aggregate sample iDNB by stage; `max` reproduces the paper pipeline."""

    if "stage" not in idnb.columns:
        raise ValueError("Stage aggregation requires a stage column")
    if method not in {"mean", "max", "median"}:
        raise ValueError(f"Unsupported aggregation method: {method}")
    result = idnb.groupby("stage", as_index=False)["idnb"].agg(method)
    return result.rename(columns={"idnb": f"idnb_{method}"})
