"""Quantum GraphQNN implementation using PennyLane.

Provides a variational circuit that maps classical features to a quantum state.
Includes utilities for generating random networks, fidelity computation,
and graph-based adjacency construction for state clustering.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility helpers – quantum versions
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> torch.Tensor:
    """Return a random unitary matrix for `num_qubits` qubits as a torch tensor."""
    dim = 2 ** num_qubits
    mat = torch.randn(dim, dim, dtype=torch.complex64)
    mat = torch.linalg.qr(mat)[0]  # orthonormalize
    return mat

def _random_qubit_state(num_qubits: int) -> torch.Tensor:
    """Return a random pure state for `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = torch.randn(dim, dtype=torch.complex64)
    vec = vec / torch.linalg.norm(vec)
    return vec

def random_training_data(target_unitary: torch.Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate pairs of input states and target states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = int(torch.log2(torch.tensor(target_unitary.shape[0])).item())
    for _ in range(samples):
        inp = _random_qubit_state(num_qubits)
        tgt = target_unitary @ inp
        dataset.append((inp, tgt))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], nn.Parameter, List[Tuple[Tensor, Tensor]], Tensor]:
    """Return a random variational network and a training set."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    # Variational parameters for the model
    params = nn.Parameter(torch.randn(1, num_qubits, 1))
    return list(qnn_arch), params, training_data, target_unitary

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two pure state vectors."""
    return float(torch.abs(torch.dot(a, b.conj())) ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
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

# --------------------------------------------------------------------------- #
# Quantum variational QNN – new
# --------------------------------------------------------------------------- #
class GraphQNNGen318(nn.Module):
    """Quantum variational graph‑QNN using PennyLane.

    Parameters
    ----------
    * qnn_arch : sequence of layer sizes.
      The final entry is the number of qubits used in the output register.
    * num_layers : number of variational blocks per layer.
    * noise_level : standard deviation of Gaussian noise added to rotation angles.
    """

    def __init__(self, qnn_arch: Sequence[int], num_layers: int = 1, noise_level: float = 0.0):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = self.qnn_arch[-1]
        self.num_layers = num_layers
        self.noise_level = noise_level
        # Parameters: shape (num_layers, num_qubits, 1)
        self.params = nn.Parameter(torch.randn(num_layers, self.num_qubits, 1))
        # Device for state evaluation
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def _circuit(self, inp: Tensor, params: Tensor) -> Tensor:
        """Return the state vector after applying the variational circuit."""
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # Encode classical input as Rx rotations
            for q, val in enumerate(inp):
                qml.RX(val, wires=q)
            # Variational layers
            for layer in range(self.num_layers):
                for q in range(self.num_qubits):
                    angle = params[layer, q, 0]
                    if self.noise_level > 0:
                        angle = angle + torch.randn_like(angle) * self.noise_level
                    qml.RY(angle, wires=q)
                # Entangling CNOT chain
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return circuit()

    def forward(self, inputs: Tensor) -> Tensor:
        """Batch‑wise forward pass returning state vectors."""
        batch_states = [self._circuit(inp, self.params) for inp in inputs]
        return torch.stack(batch_states)

    def loss(self, inputs: Tensor, targets: Tensor, threshold: float = 0.9) -> Tensor:
        """Graph‑regularised loss for quantum states."""
        outputs = self.forward(inputs)
        # Fidelity between outputs and targets
        fidelity_loss = 0.0
        for out, tgt in zip(outputs, targets):
            fid = state_fidelity(out, tgt)
            fidelity_loss += (1 - fid)
        fidelity_loss = fidelity_loss / len(outputs)
        # Graph regularisation over outputs
        n = outputs.shape[0]
        adj = torch.zeros((n, n), device=outputs.device)
        for i in range(n):
            for j in range(i + 1, n):
                fid = state_fidelity(outputs[i], outputs[j])
                if fid >= threshold:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        penalty = torch.sum(adj) / (n * (n - 1))
        return fidelity_loss + 0.1 * penalty

__all__ = [
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNGen318",
]
