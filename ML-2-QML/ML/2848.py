import torch
from torch import nn
import networkx as nx
from typing import List, Tuple, Iterable, Sequence

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a dense layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic regression pairs (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a toy dense network and training data for its final layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate a batch through the dense network, storing intermediate activations."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer.append(current)
        activations.append(layer)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Overlap between two classical activation vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph of activations with weighted edges based on fidelity."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            if j <= i:
                continue
            fid = state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

class EstimatorQNNHybrid(nn.Module):
    """Hybrid classical–quantum estimator.

    The module contains a dense neural network and a single‑qubit variational circuit.
    It can perform forward passes using either the classical network or the quantum
    estimator, controlled by the ``use_qnn`` flag.
    """

    def __init__(self, qnn_arch: Sequence[int] = (2, 8, 4, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(qnn_arch[0], qnn_arch[1]),
            nn.Tanh(),
            nn.Linear(qnn_arch[1], qnn_arch[2]),
            nn.Tanh(),
            nn.Linear(qnn_arch[2], qnn_arch[3]),
        )
        # Quantum side placeholders
        self.qc = None
        self.estimator = None

    def _build_qc(self, params: List[Parameter]) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        return qc

    def forward(self, x: Tensor, use_qnn: bool = False) -> Tensor:
        if use_qnn:
            if self.estimator is None:
                from qiskit.primitives import StatevectorEstimator
                from qiskit.circuit import Parameter
                self.estimator = StatevectorEstimator()
                params = [Parameter("p0"), Parameter("p1")]
                self.qc = self._build_qc(params)
            angle = x[0].item()
            param_dict = {self.qc.parameters[0]: angle}
            result = self.estimator(self.qc, SparsePauliOp.from_list([("Y", 1)]), parameters=param_dict)
            return torch.tensor(result[0].data.real, dtype=x.dtype)
        else:
            return self.net(x)
