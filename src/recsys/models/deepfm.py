"""LR / FM / MLP / DeepFM built from shared layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from recsys.models.layers import MLP, FieldAwareEmbedding, FMInteraction, LinearPart


class CTRModel(nn.Module):
    """Linear + optional FM + optional deep tower.

    - LR:  linear only
    - FM:  linear + FM
    - MLP: linear + deep
    - DeepFM: linear + FM + deep
    """

    def __init__(
        self,
        field_sizes: list[int],
        embedding_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        *,
        use_fm: bool = False,
        use_deep: bool = False,
    ) -> None:
        super().__init__()
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.linear = LinearPart(field_sizes)
        self.embedding = FieldAwareEmbedding(field_sizes, embedding_dim)
        self.fm = FMInteraction() if use_fm else None
        self.deep = (
            MLP(len(field_sizes) * embedding_dim, hidden_dims or [128, 64], dropout)
            if use_deep
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        embeddings = self.embedding(inputs)
        if self.fm is not None:
            logits = logits + self.fm(embeddings)
        if self.deep is not None:
            logits = logits + self.deep(embeddings.flatten(start_dim=1))
        return logits.squeeze(1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class LRModel(CTRModel):
    def __init__(self, field_sizes: list[int], **kwargs: Any) -> None:
        super().__init__(field_sizes, use_fm=False, use_deep=False, **kwargs)


class FMModel(CTRModel):
    def __init__(self, field_sizes: list[int], **kwargs: Any) -> None:
        super().__init__(field_sizes, use_fm=True, use_deep=False, **kwargs)


class MLPModel(CTRModel):
    def __init__(self, field_sizes: list[int], **kwargs: Any) -> None:
        super().__init__(field_sizes, use_fm=False, use_deep=True, **kwargs)


class DeepFM(CTRModel):
    def __init__(self, field_sizes: list[int], **kwargs: Any) -> None:
        super().__init__(field_sizes, use_fm=True, use_deep=True, **kwargs)


MODEL_REGISTRY = {
    "lr": LRModel,
    "fm": FMModel,
    "mlp": MLPModel,
    "deepfm": DeepFM,
}


def build_model(name: str, field_sizes: list[int], **kwargs: Any) -> CTRModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](field_sizes, **kwargs)
