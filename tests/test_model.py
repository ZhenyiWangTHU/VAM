import pytest
import torch

from vam.model import VAM


def test_model_shapes_and_reconstruction():
    model = VAM(4, 3, 2, embedding_dim=5)
    age, embedding, reconstructions = model(
        torch.ones(6, 4), torch.ones(6, 3), torch.ones(6, 2)
    )
    assert age.shape == (6,)
    assert embedding.shape == (6, 5)
    assert reconstructions is not None
    assert [item.shape for item in reconstructions] == [(6, 4), (6, 3), (6, 2)]


def test_model_rejects_checkpoint_dimension_mismatch():
    model = VAM(4, 3, 2)
    with pytest.raises(ValueError):
        model(torch.ones(2, 5), torch.ones(2, 3), torch.ones(2, 2))
