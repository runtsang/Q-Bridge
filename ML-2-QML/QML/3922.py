"""qml_code for UnifiedEstimatorQNN"""

import numpy as np
import networkx as nx
import itertools
from typing import List, Sequence, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import torch
from torch import nn, Tensor

# Re‑implement the small classical embedding network
class EmbeddingNet(nn.Module):
    """Tiny feed‑forward network that maps 2‑D inputs to a 1‑D embedding."""
    def __init__(self, in_features: int = 2, hidden: int = 8, out_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 8),
            nn.Tanh(),
            nn.Linear(8, out_features),
        )
        self.embedding = nn.Linear(out_features, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return self.embedding(x)

# Quantum graph‑QNN utilities
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix for `num_qubits` qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def _node_circuit(num_inputs: int,
                  input_params: List[Parameter],
                  weight_params: List[Parameter]) -> QuantumCircuit:
    """Build a simple parameterised circuit for a single node."""
    qc = QuantumCircuit(num_inputs + 1)  # output qubit appended
    for i, param in enumerate(input_params):
        qc.ry(param, i)
    for param in weight_params:
        qc.rz(param, num_inputs)
    for i in range(num_inputs):
        qc.cx(i, num_inputs)
    return qc

def _expectation_y(state: Statevector) -> float:
    """Return the expectation value of the Pauli Y observable."""
    Y = np.array([[0, -1j], [1j, 0]])
    return float(state.expectation_value(Y))

def _state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared fidelity between two pure statevectors."""
    return float(a.fidelity(b) ** 2)

def fidelity_adjacency(states: Sequence[Statevector],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities of quantum nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class QuantumGraphQNN:
    """Graph‑based quantum neural network."""
    def __init__(self,
                 architecture: Sequence[int],
                 weight_params: List[np.ndarray],
                 threshold: float = 0.8,
                 secondary: float | None = None,
                 simulator: Any = None):
        self.architecture = architecture
        self.weight_params = weight_params
        self.threshold = threshold
        self.secondary = secondary
        self.simulator = simulator if simulator is not None else AerSimulator()

    def forward(self, x: np.ndarray) -> float:
        """Run the quantum graph network on a single embedding vector."""
        num_inputs = self.architecture[0]
        input_params: List[Parameter] = [Parameter(f"inp_{i}") for i in range(num_inputs)]
        param_bindings = {p: val for p, val in zip(input_params, x)}

        node_states: List[Statevector] = []
        node_outputs: List[float] = []

        # First layer: iterate over each hidden node
        out_features = self.architecture[1]
        for node_idx in range(out_features):
            weight = self.weight_params[0][node_idx]  # 1‑D array of rotation parameters
            weight_params: List[Parameter] = [Parameter(f"w_{node_idx}_{k}") for k in range(len(weight))]
            qc = _node_circuit(num_inputs, input_params, weight_params)
            bound_qc = qc.bind_parameters(param_bindings | {p: val for p, val in zip(weight_params, weight)})
            result = self.simulator.run(bound_qc).result()
            state = Statevector(result.get_statevector(bound_qc))
            node_states.append(state)
            node_outputs.append(_expectation_y(state))

        # Build adjacency from node states
        graph = fidelity_adjacency(node_states, self.threshold,
                                   secondary=self.secondary)

        # Weighted aggregation
        if graph.number_of_edges() == 0:
            return float(np.mean(node_outputs))
        weights = np.array([sum(edge[2]['weight'] for edge in graph.edges(node, data=True))
                            for node in graph.nodes()])
        weighted_mean = np.sum(node_outputs * weights) / np.sum(weights)
        return float(weighted_mean)

class UnifiedEstimatorQNN:
    """Hybrid estimator that fuses a classical embedding network with a graph‑based quantum neural network."""
    def __init__(self,
                 architecture: Sequence[int] = (1, 2, 1),
                 simulator: Any = None):
        self.embedding = EmbeddingNet()
        # Random weight parameters for each layer of the quantum graph
        self.weight_params = [np.random.randn(out, in_) for in_, out in zip(architecture[:-1], architecture[1:])]
        self.quantum_qnn = QuantumGraphQNN(architecture,
                                           self.weight_params,
                                           simulator=simulator)

    def predict(self, x: Tensor) -> Tensor:
        """Predict the target value for a batch of inputs."""
        embed = self.embedding(x)  # (batch, 1)
        outputs = []
        for vec in embed.detach().cpu().numpy():
            out = self.quantum_qnn.forward(vec)
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)

__all__ = ["EmbeddingNet", "QuantumGraphQNN", "UnifiedEstimatorQNN"]
