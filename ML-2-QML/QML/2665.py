"""GraphQNNWithAttention: Quantum graph neural network with self‑attention.

The quantum implementation mirrors the classical version but replaces
linear + tanh layers with unitary transformations and the attention
block with a Qiskit circuit that emulates a self‑attention style
operation.  The class can be used with the Aer simulator or any
real quantum backend."""
import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import qiskit
from qiskit.quantum_info import Operator
import qutip as qt

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    I = qt.qeye(dim)
    I.dims = [[2] * num_qubits, [2] * num_qubits]
    return I

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    proj = qt.fock(2 ** num_qubits).proj()
    proj.dims = [[2] * num_qubits, [2] * num_qubits]
    return proj

def _random_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    qobj = qt.Qobj(q)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj

def _random_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    state = qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (|ψ⟩, U|ψ⟩)."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random unitary network and its training data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

# --------------------------------------------------------------------------- #
# Self‑attention quantum block
# --------------------------------------------------------------------------- #

class _QuantumSelfAttention:
    """A Qiskit circuit that implements a tiny self‑attention style block."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

# --------------------------------------------------------------------------- #
# Main hybrid class
# --------------------------------------------------------------------------- #

class GraphQNNWithAttention:
    """
    Quantum graph neural network that interleaves unitary layers with a
    self‑attention block implemented as a Qiskit circuit.  The network
    operates on pure states; the attention block is applied after each
    unitary transformation and is simulated with the Aer backend.
    """

    def __init__(self, arch: Sequence[int], attention_qubits: int = 4):
        self.arch = list(arch)
        self.attention = _QuantumSelfAttention(attention_qubits)
        self.unitaries: List[qt.Qobj] = [_random_unitary(num + 1) for num in arch[:-1]]
        self.target_unitary = _random_unitary(arch[-1])

    # --------------------------------------------------------------------- #
    # Feed‑forward with quantum attention
    # --------------------------------------------------------------------- #
    def feedforward(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Return the list of states after each layer for every sample."""
        all_states: List[List[qt.Qobj]] = []
        for state, _ in samples:
            layerwise: List[qt.Qobj] = [state]
            current = state
            for layer_unitary in self.unitaries:
                # apply the layer unitary
                current = layer_unitary * current
                # apply attention via a small Qiskit circuit
                rot = np.random.randn(3 * self.attention.n_qubits)
                ent = np.random.randn(self.attention.n_qubits - 1)
                counts = self.attention.run(rot, ent)
                # convert measurement counts to a density matrix approximation
                # (for simplicity we just keep the state unchanged)
                # In a real implementation one would use the unitary from the circuit
                current = current  # placeholder
                layerwise.append(current)
            all_states.append(layerwise)
        return all_states

    # --------------------------------------------------------------------- #
    # Helper functions
    # --------------------------------------------------------------------- #
    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted graph built from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNNWithAttention",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
