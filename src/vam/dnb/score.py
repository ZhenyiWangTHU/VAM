"""DNB module scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ssn import significant_ssn


@dataclass(frozen=True)
class DNBConfig:
    ssn_pvalue: float = 0.25
    min_hub_degree: int = 3
    two_sided: bool = False


def score_sample(
    reference_samples: np.ndarray,
    sample: np.ndarray,
    reference_correlation: np.ndarray,
    reference_mask: np.ndarray,
    feature_names: list[str] | None = None,
    config: DNBConfig | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Calculate module-level DNB scores for one sample."""

    config = config or DNBConfig()
    sample = np.asarray(sample, dtype=float)
    names = feature_names or [f"Feature_{idx}" for idx in range(sample.shape[0])]
    weights, significant = significant_ssn(
        reference_samples,
        sample,
        reference_correlation,
        reference_mask,
        config.ssn_pvalue,
        config.two_sided,
    )
    means = reference_samples.mean(axis=0)
    standard_deviations = reference_samples.std(axis=0)
    records: list[dict[str, float | str]] = []
    for hub in range(sample.shape[0]):
        neighbors = np.flatnonzero(significant[hub])
        if len(neighbors) < config.min_hub_degree:
            continue
        module = np.concatenate([[hub], neighbors])
        valid_sd = standard_deviations[module] > 0
        if not valid_sd.any():
            continue
        sd = float(
            np.mean(
                np.abs(sample[module][valid_sd] - means[module][valid_sd])
                / standard_deviations[module][valid_sd]
            )
        )
        pcc_in = float(weights[hub, neighbors].mean())
        outside_weights: list[float] = []
        module_nodes = set(module.tolist())
        for neighbor in neighbors:
            second_neighbors = np.flatnonzero(significant[neighbor])
            outside_weights.extend(
                weights[neighbor, node]
                for node in second_neighbors
                if node not in module_nodes
            )
        if not outside_weights:
            continue
        pcc_out = float(np.mean(outside_weights))
        if pcc_out <= 0:
            continue
        records.append(
            {
                "feature": names[hub],
                "dnb": sd * pcc_in / pcc_out,
                "sd": sd,
                "pcc_in": pcc_in,
                "pcc_out": pcc_out,
            }
        )
    if not records:
        return pd.DataFrame(columns=["feature", "dnb", "sd", "pcc_in", "pcc_out"])
    return pd.DataFrame(records).sort_values("dnb", ascending=False, ignore_index=True)


def score_cohort(
    reference_samples: np.ndarray,
    cohort_samples: np.ndarray,
    reference_correlation: np.ndarray,
    reference_mask: np.ndarray,
    sample_ids: list[str],
    stages: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: DNBConfig | None = None,
) -> pd.DataFrame:
    """Score a cohort and return one privacy-safe long table."""

    if len(sample_ids) != len(cohort_samples):
        raise ValueError("Sample IDs do not match cohort rows")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")

    def calculate(item: tuple[int, str, np.ndarray]) -> pd.DataFrame:
        index, sample_id, sample = item
        frame = score_sample(
            reference_samples,
            sample,
            reference_correlation,
            reference_mask,
            feature_names,
            config,
        )
        frame.insert(0, "sample_id", str(sample_id))
        if stages is not None:
            frame.insert(1, "stage", stages[index])
        return frame

    items = [
        (index, sample_id, sample)
        for index, (sample_id, sample) in enumerate(zip(sample_ids, cohort_samples))
    ]
    if n_jobs == 1:
        frames = [calculate(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            frames = list(executor.map(calculate, items))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
