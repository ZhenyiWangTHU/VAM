import numpy as np
import pandas as pd
import pytest

from vam.data import (
    align_samples,
    build_sequence_features,
    embedding_frame,
    standardize_protein_abundance,
)


def test_align_samples_uses_ids_not_row_positions():
    protein = pd.DataFrame({"p": [1, 2]}, index=["a", "b"])
    expression = pd.DataFrame({"e": [20, 10]}, index=["b", "a"])
    metabolite = pd.DataFrame({"m": [3, 4]}, index=["a", "b"])
    labels = pd.Series([40, 50], index=["a", "b"])
    aligned = align_samples(protein, expression, metabolite, labels)
    assert aligned[1].index.tolist() == ["a", "b"]
    assert aligned[1].loc["a", "e"] == 10


def test_build_sequence_features_aligns_protein_names():
    abundance = pd.DataFrame({"B": [2.0], "A": [3.0]}, index=["sample"])
    esm = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    features = build_sequence_features(abundance, esm)
    np.testing.assert_allclose(features.to_numpy(), [[3.0, 2.0]])


def test_embedding_manifest_length_is_validated():
    with pytest.raises(ValueError):
        embedding_frame(np.zeros((2, 3)), ["one"])


def test_training_statistics_can_be_reused():
    training = pd.DataFrame({"p": [1.0, 3.0]}, index=["a", "b"])
    _, means, scales = standardize_protein_abundance(training)
    external = pd.DataFrame({"p": [5.0]}, index=["c"])
    standardized, _, _ = standardize_protein_abundance(external, means, scales)
    np.testing.assert_allclose(standardized.to_numpy(), [[3.0]])
