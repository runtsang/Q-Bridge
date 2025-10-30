from __future__ import annotations

import torch
import torch.nn as nn
import networkx as nx
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
# 1. Classical fraud‑detection layer – photonic style
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool = False) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model that mirrors the photonic architecture."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2. Graph‑based utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_features, in_features, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
# 3. Unified FraudDetectionHybrid wrapper
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid:
    """
    Classical hybrid fraud‑detection module.
    Builds a photonic‑style feed‑forward network, constructs a fidelity‑based
    adjacency graph from the input data, and provides a simple
    forward pass.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vectors.
    depth : int
        Number of photonic layers to stack.
    threshold : float
        Fidelity threshold for graph construction.
    """

    def __init__(self, num_features: int, depth: int, threshold: float):
        self.num_features = num_features
        self.depth = depth
        self.threshold = threshold

        # Randomly initialise photonic layer parameters
        self.input_params = FraudLayerParameters(
            bs_theta=np.random.uniform(-np.pi, np.pi),
            bs_phi=np.random.uniform(-np.pi, np.pi),
            phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            squeeze_r=(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)),
            squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, 0.0)),
            displacement_r=(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)),
            displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            kerr=(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)),
        )
        self.layer_params = [
            FraudLayerParameters(
                bs_theta=np.random.uniform(-np.pi, np.pi),
                bs_phi=np.random.uniform(-np.pi, np.pi),
                phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                squeeze_r=(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)),
                squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                displacement_r=(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)),
                displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
                kerr=(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)),
            )
            for _ in range(depth)
        ]

        self.model = build_fraud_detection_program(self.input_params, self.layer_params)

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #

    def build_adjacency(self, data: torch.Tensor) -> nx.Graph:
        """
        Construct a weighted graph where nodes are samples and edges
        encode fidelity between their feature vectors.
        """
        return fidelity_adjacency(
            states=data,
            threshold=self.threshold,
            secondary=self.threshold * 0.5,
            secondary_weight=0.3,
        )

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Pass a batch of feature vectors through the photonic network.
        """
        return self.model(data)

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #

    def random_network(self, arch: Sequence[int], samples: int):
        """
        Generate a random classical network, training data and target weight.
        Useful for synthetic experiments.
        """
        return random_network(arch, samples)

    def feedforward(self, arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Run a classical feed‑forward neural network on a dataset.
        """
        return feedforward(arch, weights, samples)

__all__ = ["FraudDetectionHybrid"]
