import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools
from typing import Sequence, Iterable, Tuple
from dataclasses import dataclass

# --- FraudDetection utilities -------------------------------------------------
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return Layer()

    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --- GraphQNN utilities -------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: list[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> list[list[torch.Tensor]]:
    stored: list[list[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- QuantumKernelMethodGen ----------------------------------------------------
class QuantumKernelMethodGen(nn.Module):
    """Hybrid classical/graph kernel method.

    Combines an RBF kernel with optional graph‑based weighting derived from
    state fidelities.  A lightweight CNN encoder (inspired by QFCModel) is
    applied to image inputs before kernel evaluation.  The class also
    exposes utilities from GraphQNN and FraudDetection for quick data
    generation and photonic program construction.
    """

    def __init__(self, gamma: float = 1.0, use_graph: bool = False,
                 graph_threshold: float = 0.8) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_graph = use_graph
        self.graph_threshold = graph_threshold
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self._feature_dim = 16 * 7 * 7  # for 28x28 inputs

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _graph_adjacency(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        return fidelity_adjacency(states, self.graph_threshold)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel between two batches of images or feature vectors."""
        x_feat = self.cnn(x).view(x.size(0), -1)
        y_feat = self.cnn(y).view(y.size(0), -1)
        base = self._rbf(x_feat, y_feat)
        if self.use_graph:
            graph = self._graph_adjacency(torch.cat([x_feat, y_feat], dim=0))
            adj = nx.to_numpy_array(graph)
            adj = torch.from_numpy(adj).float()
            base = base * adj[:x.size(0), :x.size(0)]
        return base

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix between two lists of images."""
        a_feat = torch.stack([self.cnn(v).view(-1) for v in a])
        b_feat = torch.stack([self.cnn(v).view(-1) for v in b])
        return np.array([[self._rbf(a_feat[i], b_feat[j]).item() for j in range(len(b_feat))]
                         for i in range(len(a_feat))])

    # Expose GraphQNN utilities
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                    samples: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> nn.Sequential:
        return build_fraud_detection_program(input_params, layers)
