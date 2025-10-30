"""
Hybrid classical classifier that synthesizes the four reference architectures.

The model consists of:
1. A CNN feature extractor (QuantumNAT style).
2. A shallow auto‑encoder (Autoencoder style).
3. A fraud‑detection inspired fully‑connected stack.
4. Final 2‑class output.

The factory `build_classifier_circuit` returns the model together with
input encoding indices, cumulative parameter counts and observable indices,
mirroring the original API.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridClassifierModel(nn.Module):
    """
    Classical hybrid classifier that blends CNN, auto‑encoder and fraud‑detection
    layers.  The architecture is a direct synthesis of the four reference pairs
    and remains fully compatible with the original ``build_classifier_circuit``
    signature.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        # 1. CNN feature extractor (QuantumNAT style)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Auto‑encoder
        self.autoencoder = self._make_autoencoder(
            input_dim=16 * 7 * 7,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        # 3. Fraud‑detection inspired stack
        self.fraud_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = latent_dim if i == 0 else 4
            self.fraud_layers.append(nn.Linear(in_dim, 4))
            self.fraud_layers.append(nn.Tanh())
            self.fraud_layers.append(nn.Linear(4, 4))
        self.final = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def _make_autoencoder(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, int],
    ) -> nn.ModuleDict:
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        decoder = nn.Sequential(*decoder)
        return nn.ModuleDict({"encoder": encoder, "decoder": decoder})

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.cnn(x)
        flat = feat.view(bsz, -1)
        latent = self.autoencoder["encoder"](flat)
        out = latent
        for layer in self.fraud_layers:
            out = layer(out)
        out = self.final(out)
        out = self.bn(out)
        return out


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Factory that returns a ``HybridClassifierModel`` together with
    input encoding indices, cumulative parameter counts per layer and
    observable indices (here simply 0 and 1).
    """
    model = HybridClassifierModel(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [param.numel() for param in model.parameters()]
    observables = [0, 1]
    return model, encoding, weight_sizes, observables


__all__ = ["HybridClassifierModel", "build_classifier_circuit"]
