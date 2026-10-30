from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys.models.deepfm import build_model
from recsys.models.layers import FMInteraction


@pytest.mark.parametrize("name", ["lr", "fm", "mlp", "deepfm"])
def test_forward_shapes(name: str) -> None:
    field_sizes = [6, 4, 3]
    model = build_model(name, field_sizes, embedding_dim=8, hidden_dims=[16], dropout=0.0)
    inputs = torch.randint(0, 3, (7, len(field_sizes)))
    logits = model(inputs)
    assert logits.shape == (7,)
    assert torch.isfinite(logits).all()


def test_deepfm_has_more_parameters_than_lr() -> None:
    field_sizes = [10, 8, 6]
    lr = build_model("lr", field_sizes, embedding_dim=8)
    deepfm = build_model("deepfm", field_sizes, embedding_dim=8, hidden_dims=[16])
    assert deepfm.count_parameters() > lr.count_parameters()


def test_fm_interaction_math() -> None:
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    value = FMInteraction()(embeddings)
    # 0.5 * ((4^2 - 10) + (6^2 - 20)) = 11
    assert value.item() == pytest.approx(11.0)


def test_model_probabilities_in_range(toy_bundle) -> None:
    from recsys.serving.recommender import Recommender

    recommender = Recommender(toy_bundle)
    scores, _ = recommender.score(1, [2, 3, 4])
    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert np.isfinite(scores).all()
