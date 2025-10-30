"""Hybrid classical classifier with noise‑aware evaluation and advanced architecture."""

from __future__ import annotations

from typing import Iterable, Callable, List, Sequence

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuantumClassifierModel(nn.Module):
    """Classical neural network that mirrors a quantum classifier interface.

    The network is built with optional batch‑normalization and dropout layers,
    making it suitable for noisy evaluation and transfer to a quantum circuit.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables on batches of inputs with optional shot noise.

        Parameters
        ----------
        observables:
            Callables that map the network output to a scalar.
        parameter_sets:
            Iterable of input feature vectors.
        shots:
            If provided, Gaussian noise with std=1/√shots is added to each
            observable to emulate finite‑shot effects.
        seed:
            Random seed for reproducibility.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(next(self.parameters()).device)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().cpu().item()
                    else:
                        val = float(val)
                    row.append(val)
                if shots is not None:
                    rng = np.random.default_rng(seed)
                    row = [float(rng.normal(v, max(1e-6, 1 / shots))) for v in row]
                results.append(row)
        return results


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> tuple[nn.Module, Iterable[int], Iterable[int], List[ScalarObservable]]:
    """Build a classifier network mirroring the quantum helper interface.

    Returns a tuple of
    (network, encoding_indices, weight_sizes, observables).
    """
    net = QuantumClassifierModel(num_features, depth, dropout, use_batchnorm)
    encoding = list(range(num_features))
    weight_sizes = []
    for m in net.network.modules():
        if isinstance(m, nn.Linear):
            weight_sizes.append(m.weight.numel() + m.bias.numel())
    observables = [lambda out: out.mean(dim=-1)]
    return net, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
