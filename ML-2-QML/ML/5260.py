import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor

# ----------------------------------------------------------------------
# Classical utilities – copied and extended from GraphQNN.py
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Hybrid graph‑neural network class
# ----------------------------------------------------------------------
class GraphQNNModel(nn.Module):
    """
    Hybrid graph neural network that runs a classical feed‑forward path
    and optionally a quantum expectation head.  The architecture is
    specified by a list of layer widths.  The quantum head is supplied
    externally via a `QuantumCircuit` object that implements a `run`
    method returning expectation values.
    """
    def __init__(self, qnn_arch: Sequence[int], shift: float = 0.0,
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.classical_layers = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.classical_layers.append(nn.Linear(in_f, out_f))
        self.shift = shift
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: Tensor, quantum_circuit=None) -> Tensor:
        out = x
        for layer in self.classical_layers:
            out = torch.tanh(layer(out))
        if quantum_circuit is not None:
            # Convert to numpy, run quantum circuit
            q_in = out.detach().cpu().numpy()
            q_out = quantum_circuit.run(q_in)
            q_out = torch.tensor(q_out, device=self.device, dtype=torch.float32)
            logits = torch.sigmoid(q_out + self.shift)
            return logits
        return out

    def estimate(self, x: Tensor, quantum_circuit=None) -> Tensor:
        """
        Return a single‑value regression output.  If a quantum circuit is
        supplied, it is used as a parameterised expectation head;
        otherwise the final classical layer is used.
        """
        out = self.forward(x, quantum_circuit)
        return out

    def classical_fidelity_graph(self, states: Sequence[Tensor], threshold: float,
                                 *, secondary: float | None = None,
                                 secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    @staticmethod
    def generate_random_graph(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

__all__ = [
    "GraphQNNModel",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
