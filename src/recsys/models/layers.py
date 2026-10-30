"""Reusable embedding, linear and FM layers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FieldAwareEmbedding(nn.Module):
    """One embedding table for all fields, addressed through cumulative offsets."""

    def __init__(self, field_sizes: list[int], dim: int) -> None:
        super().__init__()
        offsets = np.concatenate([[0], np.cumsum(field_sizes)[:-1]]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.total_size = int(sum(field_sizes))
        self.embedding = nn.Embedding(self.total_size, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs + self.offsets)


class LinearPart(nn.Module):
    def __init__(self, field_sizes: list[int]) -> None:
        super().__init__()
        offsets = np.concatenate([[0], np.cumsum(field_sizes)[:-1]]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.linear = nn.Embedding(int(sum(field_sizes)), 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs + self.offsets).sum(dim=1) + self.bias


class FMInteraction(nn.Module):
    """Second-order interaction: 0.5 * ((sum e)^2 - sum e^2)."""

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        summed = embeddings.sum(dim=1)
        square_summed = (embeddings * embeddings).sum(dim=1)
        return 0.5 * (summed * summed - square_summed).sum(dim=1, keepdim=True)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
