import numpy as np
import pandas as pd

from vam.dnb import aggregate_idnb, aggregate_stages, build_reference_network
from vam.dnb.score import DNBConfig, score_sample
from vam.dnb.ssn import ssn_zscore


def test_reference_network_shapes():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(12, 5))
    correlation, mask, edges = build_reference_network(reference, pvalue=1.0)
    assert correlation.shape == (5, 5)
    assert mask.shape == (5, 5)
    assert np.array_equal(mask, mask.T)
    assert len(edges) == 10


def test_ssn_formula_matches_released_analysis():
    delta = np.array([[0.1]])
    correlation = np.array([[0.5]])
    expected = 0.1 / np.sqrt((1 - 0.5**2) / (10 - 1))
    np.testing.assert_allclose(ssn_zscore(delta, correlation, 10), [[expected]])


def test_score_sample_and_aggregation_contracts():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(20, 5))
    correlation = np.corrcoef(reference, rowvar=False)
    mask = np.ones((5, 5), dtype=bool)
    np.fill_diagonal(mask, False)
    scores = score_sample(
        reference,
        reference[0] + 3,
        correlation,
        mask,
        config=DNBConfig(ssn_pvalue=1.0, min_hub_degree=1),
    )
    assert list(scores.columns) == ["feature", "dnb", "sd", "pcc_in", "pcc_out"]

    long_scores = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "stage": [1, 1, 1, 1],
            "dnb": [4.0, 2.0, 3.0, 1.0],
        }
    )
    idnb = aggregate_idnb(long_scores, top_k=2)
    assert idnb.set_index("sample_id").loc["a", "idnb"] == 3.0
    stages = aggregate_stages(idnb, "max")
    assert stages.loc[0, "idnb_max"] == 3.0
