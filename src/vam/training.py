"""Training loop for the canonical VAM objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_checkpoint
from .data.schema import FeatureSchema
from .model import VAM


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    reconstruction_weight: float = 0.1
    l1_weight: float = 1e-3
    patience: int = 100
    seed: int = 20


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dataset(features: tuple[np.ndarray, np.ndarray, np.ndarray], ages: np.ndarray) -> TensorDataset:
    tensors = [torch.as_tensor(item, dtype=torch.float32) for item in features]
    return TensorDataset(*tensors, torch.as_tensor(ages, dtype=torch.float32))


def evaluate(
    model: VAM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sequence, expression, metabolite, ages in loader:
            prediction, _, _ = model(
                sequence.to(device), expression.to(device), metabolite.to(device), reconstruct=False
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(ages.numpy())
    pred = np.concatenate(predictions)
    true = np.concatenate(targets)
    mae = float(np.mean(np.abs(pred - true)))
    pcc = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 else float("nan")
    return {"mae": mae, "pcc": pcc}


def fit(
    train_features: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_ages: np.ndarray,
    val_features: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_ages: np.ndarray,
    schema: FeatureSchema,
    checkpoint_path: str,
    config: TrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    config = config or TrainingConfig()
    set_deterministic(config.seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VAM(*schema.model_dimensions(), embedding_dim=schema.embedding_dim).to(target)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        _dataset(train_features, train_ages),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        _dataset(val_features, val_ages), batch_size=config.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.L1Loss()
    best_mae, stale_epochs = float("inf"), 0
    best_metrics: dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        for sequence, expression, metabolite, ages in train_loader:
            inputs = tuple(item.to(target) for item in (sequence, expression, metabolite))
            optimizer.zero_grad()
            predictions, _, reconstructions = model(*inputs)
            assert reconstructions is not None
            reconstruction_loss = sum(
                loss_function(reconstructed, original)
                for reconstructed, original in zip(reconstructions, inputs)
            )
            l1_penalty = sum(parameter.abs().sum() for parameter in model.parameters())
            loss = (
                loss_function(predictions, ages.to(target))
                + config.reconstruction_weight * reconstruction_loss
                + config.l1_weight * l1_penalty
            )
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, target)
        if metrics["mae"] < best_mae:
            best_mae, stale_epochs, best_metrics = metrics["mae"], 0, metrics
            save_checkpoint(
                checkpoint_path,
                model,
                schema,
                asdict(config),
                metrics,
                epoch,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    return best_metrics
