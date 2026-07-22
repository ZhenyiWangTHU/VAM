"""Canonical VAM gated multimodal autoencoder."""

from __future__ import annotations

import torch
from torch import nn


class VAM(nn.Module):
    """Fuse three modalities with learned gates and reconstruct each input."""

    def __init__(
        self,
        sequence_dim: int,
        expression_dim: int,
        metabolite_dim: int,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        dimensions = (sequence_dim, expression_dim, metabolite_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("All input dimensions must be positive")
        self.input_dims = dimensions
        self.embedding_dim = embedding_dim
        self.gates = nn.ModuleList(nn.Linear(size, 1) for size in dimensions)
        self.encoder = nn.Linear(sum(dimensions), embedding_dim)
        self.age_head = nn.Linear(embedding_dim, 1)
        self.decoders = nn.ModuleList(nn.Linear(embedding_dim, size) for size in dimensions)

    def forward(
        self,
        sequence: torch.Tensor,
        expression: torch.Tensor,
        metabolite: torch.Tensor,
        reconstruct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        inputs = (sequence, expression, metabolite)
        for tensor, expected in zip(inputs, self.input_dims):
            if tensor.ndim != 2 or tensor.shape[1] != expected:
                raise ValueError(f"Expected input shape (batch, {expected}), got {tuple(tensor.shape)}")
        gated = [tensor * torch.sigmoid(gate(tensor)) for tensor, gate in zip(inputs, self.gates)]
        embedding = self.encoder(torch.cat(gated, dim=1))
        age = self.age_head(embedding).squeeze(-1)
        reconstructions = (
            tuple(decoder(embedding) for decoder in self.decoders) if reconstruct else None
        )
        return age, embedding, reconstructions
