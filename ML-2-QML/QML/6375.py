from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Union

import numpy as np
import qiskit
import qutip as qt
import networkx as nx

# --------------------------------------------------------------------------- #
# Utility functions for the quantum graph neural network
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator on `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector on `num_qubits` qubits."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap qubit registers `source` and `target` in `op`."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a Haar‑random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary, _ = np.linalg.qr(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Generate a Haar‑random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, unitary*state)."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Construct a random quantum graph network and training data."""
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

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Return the partial trace over all qubits *except* those in `keep`."""
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Return the partial trace over the qubits listed in `remove`."""
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply the `layer`‑th quantum channel to `input_state`."""
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
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Forward pass through the quantum graph network."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
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
# Hybrid Quantum Graph Neural Network with Quantum Self‑Attention
# --------------------------------------------------------------------------- #

class HybridQuantumGraphQNN:
    """
    Hybrid quantum graph neural network that augments each quantum layer
    with a quantum self‑attention circuit.  The self‑attention is built
    from a sequence of single‑qubit rotations followed by controlled‑RZ
    gates, mirroring the classical SelfAttention block.
    """

    def __init__(self, qnn_arch: Sequence[int], use_self_attention: bool = True):
        self.qnn_arch = list(qnn_arch)
        self.use_self_attention = use_self_attention

        # Build the underlying graph network
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(self.qnn_arch, samples=10)

        # Self‑attention parameters per layer
        if use_self_attention:
            self.rotation_params: List[np.ndarray] = [
                np.random.randn(3 * self.qnn_arch[i]) for i in range(len(self.qnn_arch) - 1)
            ]
            self.entangle_params: List[np.ndarray] = [
                np.random.randn(self.qnn_arch[i] - 1) for i in range(len(self.qnn_arch) - 1)
            ]
        else:
            self.rotation_params = None
            self.entangle_params = None

    # --------------------------------------------------------------------- #
    # Quantum self‑attention utilities
    # --------------------------------------------------------------------- #

    def _build_self_attention_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        """Return a Qiskit circuit implementing the self‑attention block."""
        n_qubits = len(rotation_params) // 3
        qr = qiskit.QuantumRegister(n_qubits, "q")
        cr = qiskit.ClassicalRegister(n_qubits, "c")
        circuit = qiskit.QuantumCircuit(qr, cr)

        # single‑qubit rotations
        for i in range(n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # controlled‑RZ entanglement
        for i in range(n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def _apply_self_attention(self, state: qt.Qobj) -> qt.Qobj:
        """Apply the self‑attention circuit to a quantum state."""
        if not self.use_self_attention:
            return state
        current = state
        for layer in range(len(self.qnn_arch) - 1):
            rot = self.rotation_params[layer]
            ent = self.entangle_params[layer]
            circuit = self._build_self_attention_circuit(rot, ent)

            backend = qiskit.Aer.get_backend("statevector_simulator")
            job = qiskit.execute(circuit, backend)
            result = job.result()
            vec = result.get_statevector(circuit)

            # Convert back to a Qobj with the same dimensionality
            current = qt.Qobj(vec.reshape(state.dims[0]), dims=state.dims)
        return current

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Forward pass through the hybrid quantum network."""
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.qnn_arch)):
                current_state = _layer_channel(self.qnn_arch, self.unitaries, layer, current_state)
                layerwise.append(current_state)

            # Apply quantum self‑attention after all layers
            current_state = self._apply_self_attention(current_state)
            layerwise.append(current_state)

            stored_states.append(layerwise)
        return stored_states

    def __repr__(self) -> str:
        return f"<HybridQuantumGraphQNN arch={self.qnn_arch} self_attn={self.use_self_attention}>"

__all__ = [
    "HybridQuantumGraphQNN",
    "_tensored_id",
    "_tensored_zero",
    "_swap_registers",
    "_random_qubit_unitary",
    "_random_qubit_state",
    "random_training_data",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
