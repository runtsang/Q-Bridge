"""GraphQNNGen452 – classical implementation.

This module unifies several lightweight models that appeared in the seed
projects:

* Random graph‑based networks (GraphQNN)
* Fraud‑detection style layers with clipping and scaling (FraudDetection)
* A tiny regression backbone (EstimatorQNN)
* A fast, shot‑noisy estimator (FastBaseEstimator)

The public API is identical to the quantum version, so the two modules
can be swapped in a hybrid workflow.

The implementation is intentionally concise but fully type‑annotated
and documented for clarity.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], torch.Tensor | float]


# --------------------------------------------------------------------------- #
# 1.  Fraud‑detection style layers
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model that mimics the photonic stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2.  Tiny regression backbone
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> nn.Module:
    """Return a small fully‑connected regressor."""

    class Estimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            return self.net(inputs)

    return Estimator()


# --------------------------------------------------------------------------- #
# 3.  Fast estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a torch model for many parameter sets and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# 4.  Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic input‑target pairs for a linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_fraud_params(num_layers: int, seed: int | None = None) -> Tuple[FraudLayerParameters, List[FraudLayerParameters]]:
    rng = random.Random(seed)
    input_params = FraudLayerParameters(
        bs_theta=rng.uniform(-np.pi, np.pi),
        bs_phi=rng.uniform(-np.pi, np.pi),
        phases=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        squeeze_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
        squeeze_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        displacement_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
        displacement_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        kerr=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
    )
    layers = [
        FraudLayerParameters(
            bs_theta=rng.uniform(-np.pi, np.pi),
            bs_phi=rng.uniform(-np.pi, np.pi),
            phases=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            squeeze_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
            squeeze_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            displacement_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
            displacement_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            kerr=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
        )
        for _ in range(num_layers)
    ]
    return input_params, layers


# --------------------------------------------------------------------------- #
# 5.  Shared class
# --------------------------------------------------------------------------- #
class GraphQNNGen452:
    """
    Unified graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[2, 4, 4, 1]``.
    use_fraud : bool, default False
        If True, the network is built from fraud‑detection style layers.
    use_estimator : bool, default False
        If True, the network is the tiny regression backbone from EstimatorQNN.
    seed : int | None, default None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        use_fraud: bool = False,
        use_estimator: bool = False,
        seed: int | None = None,
    ) -> None:
        self.arch = list(arch)
        self.seed = seed

        if use_estimator:
            self.model = EstimatorQNN()
        elif use_fraud:
            input_params, layers = random_fraud_params(len(arch) - 1, seed)
            self.model = build_fraud_detection_program(input_params, layers)
        else:
            layers = [
                nn.Linear(in_, out_)
                for in_, out_ in zip(arch[:-1], arch[1:])
            ]
            self.model = nn.Sequential(*layers)

    # --------------------------------------------------------------------- #
    # 5.1  Random network generation
    # --------------------------------------------------------------------- #
    def random_network(self, samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random target (weight matrix) and synthetic dataset.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return self.arch, weights, training_data, target_weight

    # --------------------------------------------------------------------- #
    # 5.2  Feed‑forward
    # --------------------------------------------------------------------- #
    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """
        Run the model on the provided samples and collect activations.

        Parameters
        ----------
        samples : Iterable[Tuple[Tensor, Tensor]]
            Each element is ``(input, target)``; the target is unused.
        """
        stored: List[List[Tensor]] = []
        for inp, _ in samples:
            activations = [inp]
            current = inp
            for layer in self.model:
                current = layer(current)
                activations.append(current)
            stored.append(activations)
        return stored

    # --------------------------------------------------------------------- #
    # 5.3  Fidelity helpers
    # --------------------------------------------------------------------- #
    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Squared overlap of two normalized vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # 5.4  Estimator evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Fast evaluation using FastEstimator."""
        estimator = FastEstimator(self.model)
        return estimator.evaluate(observables, parameter_sets)

    def add_noise(
        self,
        results: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to the results."""
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "GraphQNNGen452",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "EstimatorQNN",
    "FastBaseEstimator",
    "FastEstimator",
]
