"""Enhanced classical sampler network with residual connections and regularisation.

The class mirrors the original SamplerQNN interface but adds depth, batch‑normalisation,
dropout and a skip‑connection to improve expressivity and mitigate over‑fitting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen346(nn.Module):
    """
    A more expressive sampler network.

    Architecture:
        Input (2) -> Linear(2→8) -> BatchNorm1d -> ReLU
        -> Linear(8→8) -> BatchNorm1d -> ReLU
        -> Dropout(0.2)
        -> Linear(8→2)
        -> Softmax

    A residual connection adds the input to the output before softmax,
    allowing the network to learn identity mappings more easily.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Ensure input shape is (batch, 2)
        out = self.net(inputs)
        # Residual skip: add the original input (broadcasted) to logits
        out = out + inputs
        return F.softmax(out, dim=-1)


__all__ = ["SamplerQNNGen346"]
