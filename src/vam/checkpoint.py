"""Versioned VAM checkpoint serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .data.schema import FeatureSchema
from .model import VAM


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: VAM,
    schema: FeatureSchema,
    training_config: dict[str, Any],
    metrics: dict[str, float],
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "schema": asdict(schema),
            "training_config": training_config,
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[VAM, FeatureSchema, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')}")
    schema = FeatureSchema(**payload["schema"])
    model = VAM(*schema.model_dimensions(), embedding_dim=schema.embedding_dim)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, schema, payload
