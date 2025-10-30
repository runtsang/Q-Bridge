"""Hybrid classical classifier mirroring the quantum interface.

The model combines a parameterized fully‑connected layer, a small
sampler network, and a residual feed‑forward backbone.  It exposes
the same build interface as the original QuantumClassifierModel,
returning a torch.nn.Module together with encoding metadata,
parameter counts per layer, and observable callables.  A lightweight FastEstimator
wrapper adds Gaussian shot noise to emulate quantum measurement
uncertainty.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Classical building blocks
# --------------------------------------------------------------------------- #

class FCLayer(nn.Module):
    """Simple fully‑connected layer with tanh activation."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class SamplerQNNModule(nn.Module):
    """Soft‑max sampler inspired by the Qiskit SamplerQNN example."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
#  Hybrid classifier
# --------------------------------------------------------------------------- #

class QuantumClassifierModel(nn.Module):
    """Residual feed‑forward network that mirrors the quantum interface."""

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        layers: List[nn.Module] = []

        # 1. Parameterised fully‑connected layer (FCL)
        layers.append(FCLayer(num_features))

        # 2. Sampler head
        layers.append(SamplerQNNModule())

        # 3. Residual backbone
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features

        # 4. Output head
        layers.append(nn.Linear(in_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --------------------------------------------------------------------------- #
#  Estimator with optional shot noise
# --------------------------------------------------------------------------- #

class FastEstimator:
    """Wraps a torch model and optionally adds Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables: list of callables mapping network output to a scalar
        parameter_sets: batched parameter tensors
        shots: if provided, Gaussian noise with variance 1/shots is added
        seed: seed for reproducibility
        """
        observables = list(observables)
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)

        return noisy


# --------------------------------------------------------------------------- #
#  Build interface mirroring the quantum helper
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], List[int], List[Callable[[torch.Tensor], torch.Tensor]]]:
    """
    Construct a hybrid classifier and return network, encoding indices,
    parameter counts per layer, and observable callables.

    Parameters
    ----------
    num_features: number of input features / qubits
    depth: number of residual layers
    """
    network = QuantumClassifierModel(num_features, depth)

    # Encoding indices correspond to the features fed into the network
    encoding = list(range(num_features))

    # Count parameters per linear layer (including FCL and output head)
    weight_sizes: List[int] = []
    for m in network.network:
        if isinstance(m, nn.Linear):
            weight_sizes.append(m.weight.numel() + m.bias.numel())

    # Observables: extract two scalar outputs from the final layer
    observables = [
        lambda out: out[:, 0],
        lambda out: out[:, 1],
    ]

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "QuantumClassifierModel", "FastEstimator"]
