import numpy as np

from vam.data.schema import FeatureSchema
from vam.inference import predict
from vam.training import TrainingConfig, fit


def test_train_checkpoint_and_infer(tmp_path):
    rng = np.random.default_rng(3)
    features = (
        rng.normal(size=(20, 4)).astype("float32"),
        rng.normal(size=(20, 3)).astype("float32"),
        rng.normal(size=(20, 2)).astype("float32"),
    )
    ages = (45 + features[0][:, 0]).astype("float32")
    schema = FeatureSchema(
        [f"s{index}" for index in range(4)],
        [f"e{index}" for index in range(3)],
        ["m0", "m1"],
        embedding_dim=5,
    )
    checkpoint = tmp_path / "vam.pt"
    metrics = fit(
        tuple(item[:15] for item in features),
        ages[:15],
        tuple(item[15:] for item in features),
        ages[15:],
        schema,
        str(checkpoint),
        TrainingConfig(epochs=2, batch_size=5, patience=2, l1_weight=0.0),
        device="cpu",
    )
    assert checkpoint.exists()
    assert "mae" in metrics
    predicted, embeddings = predict(
        str(checkpoint), tuple(item[15:] for item in features), device="cpu"
    )
    assert predicted.shape == (5,)
    assert embeddings.shape == (5, 5)
