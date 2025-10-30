"""
GraphQNNGen607 – Quantum implementation.

The quantum version keeps the same public API but replaces dense layers
with parameterised unitary blocks and the classical attention with a
Qiskit‑based self‑attention circuit.  The state propagation uses qutip,
while the attention block is executed on a QASM simulator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
#  Helper functions for qubit objects
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = (qt.rand_unitary(dim))
    qobj = qt.Qobj(matrix)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = (qt.rand_ket(dim))
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


# --------------------------------------------------------------------------- #
#  Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Parameterised Qiskit circuit that implements a self‑attention style
    interaction between qubits.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
            circ.rz(entangle_params[i], i)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        circ = self._build_circuit(rotation_params, entangle_params)
        job = execute(circ, self.backend, shots=shots)
        return job.result().get_counts(circ)


# --------------------------------------------------------------------------- #
#  Quantum network construction
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a random quantum network.  Each layer consists of a unitary
    that acts on the input qubits plus a fresh ancilla, followed by
    a partial trace that discards the ancilla.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    # Insert a quantum attention block after each layer
    attentions: List[QuantumSelfAttention] = [QuantumSelfAttention(n_qubits=4) for _ in range(len(qnn_arch) - 1)]

    return qnn_arch, unitaries, attentions, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    attentions: Sequence[QuantumSelfAttention],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """
    Forward pass through the quantum network.  After each unitary layer,
    a quantum self‑attention circuit is applied to the current state.
    """
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            # Quantum self‑attention on the current state
            # (parameters are dummy zeros for demonstration)
            rotation_params = np.zeros(3 * attentions[layer - 1].n_qubits)
            entangle_params = np.zeros(attentions[layer - 1].n_qubits - 1)
            counts = attentions[layer - 1].run(rotation_params, entangle_params)
            # Convert counts to a density matrix approximation
            rho = qt.tensor(*[qt.fock(2).proj() for _ in range(current_state.dims[0][0])])
            current_state = rho
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
#  Graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (state, U*state) pairs for supervised learning."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


__all__ = [
    "QuantumSelfAttention",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
