"""
GraphQNN__gen022.py

Quantum counterpart of the hybrid module.  Builds a parameterised
ansatz that can be trained with Pennylane and optionally exported to
Qiskit.  The interface mirrors the classical module and adds
functions for converting between flat tensors and Qobj matrices.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import scipy as sc
import qutip as qt

# ---------- Utilities ----------
def _tensor_to_unitary(tensor: np.ndarray) -> np.ndarray:
    """Convert a flat vector into a unitary matrix via QR."""
    dim = int(np.sqrt(tensor.size))
    q, _ = np.linalg.qr(tensor.reshape(dim, dim))
    return q

def _unitary_to_tensor(unitary: np.ndarray) -> np.ndarray:
    return unitary.flatten()

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(mat)
    return q

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim,)) + 1j * sc.random.normal(size=(dim,))
    vec /= np.linalg.norm(vec)
    return vec

# ---------- Ansatz ----------
def _build_ansatz(num_qubits: int, depth: int) -> qml.QNode:
    """Return a Pennylane QNode implementing a Trotter‑like ansatz."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params: np.ndarray, state: np.ndarray):
        qml.StatePrep(state, wires=range(num_qubits))
        idx = 0
        for d in range(depth):
            for i in range(num_qubits):
                qml.RX(params[idx], wires=i); idx += 1
                qml.RY(params[idx], wires=i); idx += 1
                qml.RZ(params[idx], wires=i); idx += 1
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()
    return circuit

# ---------- Classical equivalents ----------
def random_training_data(unitary: np.ndarray, samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training data for the ansatz."""
    dataset = []
    dim = unitary.shape[0]
    num_qubits = int(np.log2(dim))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target_state = unitary @ state
        dataset.append((state, target_state))
    return dataset

def random_network(qnn_arch: list[int], depth: int, samples: int):
    """Return architecture, ansatz, training data and target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    ansatz = _build_ansatz(qnn_arch[-1], depth)
    return qnn_arch, ansatz, training_data, target_unitary

# ---------- Feedforward ----------
def feedforward(qnn_arch: Sequence[int], ansatz: qml.QNode, samples: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[list[np.ndarray]]:
    """Run the ansatz on each sample and store the state after each layer."""
    stored_states = []
    for state, _ in samples:
        # Random parameters for demonstration
        num_params = 3 * qnn_arch[-1] * ansatz.__closure__[0].cell_contents.depth
        params = np.random.normal(size=num_params)
        final_state = ansatz(params, state)
        stored_states.append([state, final_state])
    return stored_states

# ---------- Fidelity helpers ----------
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap squared between pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---------- Hybrid class ----------
class GraphQNN__gen022:
    """Quantum implementation of the hybrid graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture of the quantum network (number of qubits per layer).
    depth : int, optional
        Depth of the parameterised ansatz.
    """
    def __init__(self, arch: Sequence[int], depth: int = 2, device: str | None = None):
        self.arch = tuple(arch)
        self.depth = depth
        self.device = device or "default.qubit"
        self.ansatz = _build_ansatz(self.arch[-1], self.depth)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], depth: int, samples: int):
        """Return architecture, ansatz, training data and target unitary."""
        return random_network(qnn_arch, depth, samples)

    def predict(self, params: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Evaluate the ansatz with given parameters and input state."""
        return self.ansatz(params, state)

    def train_ansatz(self, dataset: list[tuple[np.ndarray, np.ndarray]], lr: float = 0.01, epochs: int = 200):
        """Simple gradient‑descent training of the ansatz."""
        num_params = 3 * self.arch[-1] * self.depth
        params = np.random.normal(size=num_params)
        for _ in range(epochs):
            loss = 0.0
            grads = np.zeros_like(params)
            for state, target in dataset:
                pred = self.ansatz(params, state)
                loss += pnp.mean(pnp.abs(pred - target) ** 2)
                grad_fn = qml.grad(lambda p: pnp.mean(pnp.abs(self.ansatz(p, state) - target) ** 2))
                grads += grad_fn(params)
            loss /= len(dataset)
            grads /= len(dataset)
            params -= lr * grads
        self.params = params

    def get_qiskit_circuit(self) -> "qiskit.QuantumCircuit":
        """Return a Qiskit circuit that implements the trained ansatz."""
        try:
            import qiskit
        except ImportError:
            raise ImportError("qiskit is required to export the circuit")
        if not hasattr(self, "params"):
            raise RuntimeError("Ansatz has not been trained yet")
        qc = qiskit.QuantumCircuit(self.arch[-1])
        idx = 0
        for d in range(self.depth):
            for i in range(self.arch[-1]):
                qc.rx(self.params[idx], i); idx += 1
                qc.ry(self.params[idx], i); idx += 1
                qc.rz(self.params[idx], i); idx += 1
            for i in range(self.arch[-1] - 1):
                qc.cx(i, i + 1)
        return qc

__all__ = [
    "GraphQNN__gen022",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
