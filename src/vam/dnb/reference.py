"""Reference correlation network construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def build_reference_network(
    reference_samples: np.ndarray,
    feature_names: list[str] | None = None,
    pvalue: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the Pearson reference network from samples x features."""

    values = np.asarray(reference_samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Reference data must contain at least three samples")
    names = feature_names or [f"Feature_{idx}" for idx in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise ValueError("Feature names do not match the reference matrix")
    correlation = np.corrcoef(values, rowvar=False)
    mask = np.zeros_like(correlation, dtype=bool)
    edges: list[dict[str, object]] = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            coefficient, edge_pvalue = stats.pearsonr(values[:, left], values[:, right])
            if np.isfinite(edge_pvalue) and edge_pvalue < pvalue:
                mask[left, right] = mask[right, left] = True
                edges.append(
                    {
                        "feature_1": names[left],
                        "feature_2": names[right],
                        "correlation": coefficient,
                        "pvalue": edge_pvalue,
                    }
                )
    return correlation, mask, pd.DataFrame(edges)
