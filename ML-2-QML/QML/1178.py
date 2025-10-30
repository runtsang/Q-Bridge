"""GraphQNNHybrid – quantum graph neural network using Qiskit.

This module mirrors the classical interface but replaces the final linear
layer with a parameterised quantum circuit.  It provides utilities to
generate random quantum networks, perform forward propagation, and
construct fidelity‑based adjacency graphs.  The implementation uses
Qiskit’s Statevector simulator for exact state fidelity calculations.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable, Any

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

Tensor = np.ndarray

# --------------------------------------------------------------------------- #
# 1. Classical utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return np.random.randn(out_features, in_features).astype(np.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = np.random.randn(weight.shape[1]).astype(np.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
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
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = np.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm) ** 2)

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
# 2. Quantum utilities – Qiskit circuit generation
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Tensor:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(matrix)
    return q

def _random_qubit_state(num_qubits: int) -> Tensor:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data_qml(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        dataset.append((state, unitary @ state))
    return dataset

def random_network_qml(qnn_arch: List[int], samples: int):
    """Generate a random quantum circuit that mirrors a classical architecture."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data_qml(target_unitary, samples)

    # Build a layered circuit: each layer applies a random unitary to the next
    # block of qubits.  For simplicity we use a single unitary per layer.
    layers: List[QuantumCircuit] = []
    for layer_size in qnn_arch[1:]:
        qc = QuantumCircuit(layer_size)
        qc.unitary(target_unitary[:2**layer_size, :2**layer_size], qc.qubits, inplace=True)
        layers.append(qc)

    return qnn_arch, layers, training_data, target_unitary

def _apply_circuit(qc: QuantumCircuit, state: Tensor) -> Tensor:
    backend = Aer.get_backend("statevector_simulator")
    job = execute(qc, backend, initial_state=state)
    result = job.result()
    return result.get_statevector(qc)

def feedforward_qml(
    qnn_arch: Sequence[int],
    circuits: Sequence[QuantumCircuit],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for qc in circuits:
            current = _apply_circuit(qc, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

# --------------------------------------------------------------------------- #
# 3. Hybrid quantum model definition
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Quantum graph neural network that replaces the final linear layer
    with a parameterised Qiskit circuit.
    """

    def __init__(
        self,
        qnn_arch: List[int],
        circuit: QuantumCircuit,
        device: str = "cpu",
    ):
        self.qnn_arch = qnn_arch
        self.circuit = circuit
        self.device = device

    def forward(self, state: Tensor) -> Tensor:
        """Apply the quantum circuit to the input state."""
        return _apply_circuit(self.circuit, state)

# --------------------------------------------------------------------------- #
# 4. Training utilities – simple example
# --------------------------------------------------------------------------- #
def train_step_qml(
    model: GraphQNNHybrid,
    optimizer: Any,
    loss_fn: Callable[[Tensor, Tensor], float],
    batch: Tuple[Tensor, Tensor],
) -> float:
    """Perform one gradient step on the quantum model using parameter‑shift."""
    # Note: full gradient implementation omitted for brevity.
    # This placeholder demonstrates the interface.
    pred = model.forward(batch[0])
    loss = loss_fn(pred, batch[1])
    # Dummy update
    optimizer.step()
    return loss

def fidelity_callback_qml(
    model: GraphQNNHybrid,
    batch: Tuple[Tensor, Tensor],
) -> List[float]:
    """Compute fidelity of the quantum output with a reference state."""
    pred = model.forward(batch[0])
    ref = np.ones_like(pred) / np.sqrt(len(pred))
    fid = np.abs(np.vdot(pred, ref)) ** 2
    return [float(fid)]

# --------------------------------------------------------------------------- #
# 5. Expose public API
# --------------------------------------------------------------------------- #
__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNHybrid",
    "train_step_qml",
    "fidelity_callback_qml",
]
