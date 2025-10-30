import pennylane as qml
import torch
import networkx as nx
import numpy as np
from typing import Sequence, Tuple

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Quantum graph layer using PennyLane
# --------------------------------------------------------------------------- #
class QuantumGraphLayer:
    """
    Variational layer that encodes a graph structure into a quantum
    circuit.  Each node is represented by a qubit; edges are
    encoded via controlled‑rotation gates.  Parameters are shared
    across layers to reduce the number of trainable weights.
    """
    def __init__(self, num_qubits: int, num_layers: int = 2, dev: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(dev, wires=num_qubits)

        # Trainable parameters: rotation angles for each layer
        self.params = torch.nn.Parameter(
            torch.randn(num_layers, num_qubits, 3, dtype=torch.float32)
        )

    def _encode_features(self, features: Tensor):
        """Encode classical node features into qubit states via RY rotations."""
        for i, feat in enumerate(features):
            qml.RY(feat.item(), wires=i)

    def _apply_edge_gates(self, edges: Sequence[Tuple[int, int]]):
        """Apply controlled‑rotation gates along graph edges."""
        for src, tgt in edges:
            qml.CRX(np.pi / 4, wires=[src, tgt])

    def circuit(self, features: Tensor, edges: Sequence[Tuple[int, int]]):
        self._encode_features(features)
        for layer in range(self.num_layers):
            self._apply_edge_gates(edges)
            # Parameterized single‑qubit rotations
            for q in range(self.num_qubits):
                qml.Rot(
                    self.params[layer, q, 0].item(),
                    self.params[layer, q, 1].item(),
                    self.params[layer, q, 2].item(),
                    wires=q,
                )
        return qml.expval(qml.PauliZ(0))  # Observable for output

    def __call__(self, features: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:
        qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        return qnode(features, edges)

# --------------------------------------------------------------------------- #
# Hybrid GraphQNN combining classical GCN and quantum layer
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Hybrid architecture: a classical GCN followed by a quantum graph
    layer.  The quantum layer is trained to reproduce the output of
    the GCN, measured via a fidelity loss.
    """
    def __init__(self, in_features: int, hidden: int, out_features: int, num_qubits: int):
        self.cnn = ClassicalGCN(in_features, hidden, out_features)
        self.qlayer = QuantumGraphLayer(num_qubits, num_layers=2)

    def forward(self, data):
        """
        Forward pass: compute classical output and quantum state.
        """
        # Classical GCN output
        classical_out = self.cnn(data)

        # Prepare quantum inputs
        features = data.x.squeeze()  # shape: [num_nodes, in_features]
        # Flatten node features to a single scalar per node for encoding
        node_features = torch.mean(features, dim=1)
        edges = list(zip(*data.edge_index.cpu().tolist()))

        # Quantum output
        quantum_out = self.qlayer(node_features, edges)

        return classical_out, quantum_out

    def loss(self, quantum_out: Tensor, classical_out: Tensor):
        """
        Fidelity loss between quantum and classical outputs.
        """
        return quantum_fidelity_loss(quantum_out, classical_out)

# --------------------------------------------------------------------------- #
# Random data generation utilities (quantum)
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int):
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q

def random_training_data(unitary: np.ndarray, samples: int):
    dataset = []
    for _ in range(samples):
        state = np.random.randn(unitary.shape[0]) + 1j * np.random.randn(unitary.shape[0])
        state /= np.linalg.norm(state)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Generate a random hybrid network: a list of weight matrices for the
    classical GCN and a target unitary for the quantum layer.
    """
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), weights, training_data, target_unitary

# --------------------------------------------------------------------------- #
# Fidelity‑based graph utilities
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "QuantumGraphLayer",
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
