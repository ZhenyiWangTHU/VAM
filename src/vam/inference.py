"""Checkpoint-backed VAM inference."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import load_checkpoint


def predict(
    checkpoint_path: str,
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, schema, _ = load_checkpoint(checkpoint_path, target)
    expected = schema.model_dimensions()
    arrays = tuple(np.asarray(item, dtype=np.float32) for item in features)
    row_counts = {array.shape[0] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError("All modalities must contain the same number of samples")
    for array, dimension in zip(arrays, expected):
        if array.ndim != 2 or array.shape[1] != dimension:
            raise ValueError(f"Checkpoint expects feature dimension {dimension}, got {array.shape}")
    loader = DataLoader(
        TensorDataset(*(torch.from_numpy(item) for item in arrays)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    ages, embeddings = [], []
    with torch.no_grad():
        for sequence, expression, metabolite in loader:
            prediction, embedding, _ = model(
                sequence.to(target),
                expression.to(target),
                metabolite.to(target),
                reconstruct=False,
            )
            ages.append(prediction.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(ages), np.concatenate(embeddings)
