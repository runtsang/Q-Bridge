"""UnifiedSelfAttentionQNN – quantum component.

The module implements a variational self‑attention circuit that
produces quantum statevectors.  It also provides graph‑based utilities
for state fidelity and adjacency, mirroring the classical
GraphQNN functions.  The design is inspired by
the quantum SelfAttention and GraphQNN utilities.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer

Tensor = np.ndarray  # use numpy arrays for parameters

# --------------------------------------------------------------------------- #
# Random quantum network generator
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    rand_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(rand_matrix)  # QR decomposition yields a unitary
    return q

def random_network(qnn_arch: Sequence[int], samples: int = 10):
    """Generate a random quantum neural network architecture.

    Returns the architecture list, a list of lists of unitary matrices
    (one per layer), training data samples, and the target unitary.
    """
    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(num_outputs):
            op = _random_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = []
    for _ in range(samples):
        state = np.random.randn(2 ** qnn_arch[-1]) + 1j * np.random.randn(2 ** qnn_arch[-1])
        state /= np.linalg.norm(state)
        training_data.append((state, target_unitary @ state))
    return list(qnn_arch), unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute squared absolute overlap between two statevectors."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Feedforward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Propagate each sample through the quantum layers."""
    states_list = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            # apply first unitary of the layer
            op = unitaries[layer][0]
            current = op @ current
            # if more than one output, apply additional unitaries
            for extra in unitaries[layer][1:]:
                current = extra @ current
            layerwise.append(current)
        states_list.append(layerwise)
    return states_list

# --------------------------------------------------------------------------- #
# UnifiedSelfAttentionQNN quantum class
# --------------------------------------------------------------------------- #
class UnifiedSelfAttentionQNN:
    """Variational quantum self‑attention circuit.

    The circuit applies a sequence of rotation gates followed by
    controlled‑rotation entangling gates.  It is designed to be
    compatible with the classical UnifiedSelfAttentionQNN interface
    while providing genuine quantum state‑vector outputs.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """Execute the variational circuit on the given backend."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

    def statevector(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """Return the statevector produced by the circuit."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        backend = Aer.get_backend("statevector_simulator")
        job = qiskit.execute(circuit, backend)
        result = job.result()
        state = result.get_statevector(circuit)
        return np.array(state)

# --------------------------------------------------------------------------- #
# Compatibility wrapper
# --------------------------------------------------------------------------- #
def SelfAttention() -> UnifiedSelfAttentionQNN:
    """Return a pre‑configured UnifiedSelfAttentionQNN quantum instance."""
    return UnifiedSelfAttentionQNN(n_qubits=4)

__all__ = [
    "UnifiedSelfAttentionQNN",
    "SelfAttention",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
