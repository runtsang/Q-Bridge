"""Hybrid classical self‑attention that optionally incorporates auto‑encoding and fraud‑detection layers.

The class is designed to replace the original SelfAttention helper while providing
additional preprocessing stages and a convenient estimator interface.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, List
from.Autoencoder import Autoencoder
from.FraudDetection import FraudLayerParameters
from.FastBaseEstimator import FastEstimator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _make_fraud_layer(params: FraudLayerParameters) -> nn.Module:
    """Build a simple 2‑D linear layer with parameters inspired by the photonic fraud circuit."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    return linear

class HybridSelfAttention:
    """Hybrid classical self‑attention block."""
    def __init__(self,
                 embed_dim: int,
                 autoencoder: nn.Module | None = None,
                 fraud_params: FraudLayerParameters | None = None):
        self.embed_dim = embed_dim
        self.autoencoder = autoencoder
        self.fraud_layer = _make_fraud_layer(fraud_params) if fraud_params else None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.to(self._device)
        if self.autoencoder:
            with torch.no_grad():
                x = self.autoencoder.encode(x)
        if self.fraud_layer:
            x = self.fraud_layer(x)
        return x

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        x = self._preprocess(inp)
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        query = torch.matmul(x, rot)
        key   = torch.matmul(x, ent)
        scores = torch.softmax(torch.matmul(query, key.T) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, x).cpu().numpy()

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        estim = FastEstimator(self)
        return estim.evaluate(observables, parameter_sets)

def HybridSelfAttentionFactory(embed_dim: int,
                               autoencoder_config: dict | None = None,
                               fraud_params: FraudLayerParameters | None = None) -> HybridSelfAttention:
    autoencoder = Autoencoder(**autoencoder_config) if autoencoder_config else None
    return HybridSelfAttention(embed_dim, autoencoder=autoencoder, fraud_params=fraud_params)

__all__ = ["HybridSelfAttention", "HybridSelfAttentionFactory"]
