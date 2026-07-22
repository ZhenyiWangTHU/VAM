"""Sample-specific network statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ssn_zscore(
    delta_correlation: np.ndarray,
    reference_correlation: np.ndarray,
    reference_size: int,
) -> np.ndarray:
    if reference_size < 2:
        raise ValueError("At least two reference samples are required")
    clipped = np.clip(reference_correlation, -0.99999999, 0.99999999)
    denominator = np.sqrt((1.0 - np.square(clipped)) / (reference_size - 1))
    return np.divide(
        delta_correlation,
        denominator,
        out=np.zeros_like(delta_correlation),
        where=denominator > 0,
    )


def significant_ssn(
    reference_samples: np.ndarray,
    sample: np.ndarray,
    reference_correlation: np.ndarray,
    reference_mask: np.ndarray,
    pvalue: float = 0.25,
    two_sided: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute SSN edge weights and their significance mask.

    The default one-sided tail reproduces the original VAM analysis.
    """

    combined = np.vstack([reference_samples, np.asarray(sample)[None, :]])
    sample_correlation = np.corrcoef(combined, rowvar=False)
    delta = sample_correlation - reference_correlation
    zscore = ssn_zscore(delta, reference_correlation, len(reference_samples))
    tail = 2.0 * (1.0 - norm.cdf(np.abs(zscore))) if two_sided else 1.0 - norm.cdf(
        np.abs(zscore)
    )
    significant = (tail < pvalue) & reference_mask
    np.fill_diagonal(significant, False)
    return np.abs(delta), significant
