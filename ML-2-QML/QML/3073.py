"""GraphQNNGen348: Quantum implementation with self‑attention circuits and fidelity‑based graph construction."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info import Statevector, Operator

# --------------------------------------------------------------------------- #
#  Utility helpers – qubit identities, zero projectors, and unitary generators
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> Operator:
    """Identity operator of size 2**num_qubits."""
    return Operator(np.eye(2 ** num_qubits))

def _tensored_zero(num_qubits: int) -> Operator:
    """Zero projector on num_qubits qubits."""
    return Operator(np.diag([1] + [0] * (2 ** num_qubits - 1)))

def _random_qubit_unitary(num_qubits: int) -> Operator:
    """Random unitary on num_qubits qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)  # QR decomposition gives a random unitary
    return Operator(q)

def _random_qubit_state(num_qubits: int) -> Statevector:
    """Random pure state on num_qubits qubits."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

# --------------------------------------------------------------------------- #
#  Quantum self‑attention circuit builder
# --------------------------------------------------------------------------- #

class QuantumSelfAttention:
    """Self‑attention style block implemented as a parameterised quantum circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("unitary_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling controlled‑X gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def unitary(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> Operator:
        """Return the unitary matrix corresponding to the attention circuit."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        unitary = self.backend.run(circuit).result().get_unitary()
        return Operator(unitary)

# --------------------------------------------------------------------------- #
#  Random training data and network generator
# --------------------------------------------------------------------------- #

def random_training_data(unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate input/target pairs using a fixed target unitary."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    num_qubits = int(np.log2(unitary.dim))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = state.evolve(unitary)
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Sample a random QNN architecture with attention‑based unitaries."""
    attention = QuantumSelfAttention(n_qubits=max(qnn_arch))
    unitaries: List[List[Operator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_qubits = qnn_arch[layer - 1]
        out_qubits = qnn_arch[layer]
        layer_ops: List[Operator] = []
        for _ in range(out_qubits):
            # Random parameters for the attention circuit
            rot = np.random.uniform(-np.pi, np.pi, size=3 * in_qubits)
            ent = np.random.uniform(-np.pi, np.pi, size=max(in_qubits - 1, 1))
            unitary = attention.unitary(rot, ent)
            layer_ops.append(unitary)
        unitaries.append(layer_ops)

    target_unitary = unitaries[-1][0]  # take first output unitary as target
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
#  Partial trace utilities for state propagation
# --------------------------------------------------------------------------- #

def _partial_trace(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the reduced statevector over the qubits in *keep*."""
    return state.partial_trace(keep)

# --------------------------------------------------------------------------- #
#  Feed‑forward propagation through the QNN
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Operator]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate each sample through the quantum network."""
    stored_states: List[List[Statevector]] = []
    for state, _ in samples:
        layerwise: List[Statevector] = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            # Apply each output unitary and keep the corresponding qubits
            output_ops = unitaries[layer]
            # For simplicity, concatenate outputs into a single state
            new_state = None
            for idx, op in enumerate(output_ops):
                out_state = current.evolve(op)
                if new_state is None:
                    new_state = out_state
                else:
                    new_state = Statevector(np.kron(new_state.data, out_state.data))
            # Keep only the output qubits
            keep = list(range(qnn_arch[layer]))
            current = _partial_trace(new_state, keep)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

# --------------------------------------------------------------------------- #
#  Fidelity utilities – quantum state overlap and graph construction
# --------------------------------------------------------------------------- #

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure statevectors."""
    return abs(np.vdot(a.data, b.data)) ** 2

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Weighted adjacency graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuantumSelfAttention",
]
