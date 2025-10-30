"""Quantum graph neural network with an optional quanvolution front‑end.

This module combines the graph‑based quantum feed‑forward utilities
from the original QML seed with a quantum convolutional filter
(“quanvolution”) that can be used as a drop‑in replacement for the
classical Conv filter.  The class GraphQNNGen220 exposes a unified
interface that mirrors the classical version while remaining fully
quantum‑centric.

Key features
------------
* Random network generation with parameterised unitary layers.
* Quanvolution filter implemented with Qiskit.
* Fidelity‑based adjacency graph construction.
* State‑fidelity computation for pure quantum states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
#  Core quantum utilities
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


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random quantum graph network and training data."""
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
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
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
    """Run a forward pass through the quantum graph network."""
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


# --------------------------------------------------------------------------- #
#  Quanvolution filter
# --------------------------------------------------------------------------- #

class QuanvCircuit:
    """Quantum convolutional filter implemented with Qiskit."""

    def __init__(
        self,
        kernel_size: int,
        backend: qiskit.providers.Backend,
        shots: int,
        threshold: float,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(
            self.n_qubits, depth=2, measure=False
        )
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array of shape (kernel_size, kernel_size)."""
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
#  Unified quantum graph neural network
# --------------------------------------------------------------------------- #

class GraphQNNGen220:
    """
    Quantum graph neural network with an optional quanvolution front‑end.
    The API mirrors the classical GraphQNNGen220 to enable side‑by‑side
    experimentation.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        kernel_size: int = 2,
        use_quanv: bool = True,
        quanv_threshold: float = 127,
        shots: int = 100,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.kernel_size = kernel_size
        self.use_quanv = use_quanv

        if use_quanv:
            backend = qiskit.Aer.get_backend("qasm_simulator")
            self.quanv = QuanvCircuit(
                kernel_size,
                backend,
                shots=shots,
                threshold=quanv_threshold,
            )
        else:
            self.quanv = None

    def random_network(self, samples: int):
        """Generate a random quantum network and training data."""
        return random_network(self.qnn_arch, samples)

    def random_training_data(self, unitary: qt.Qobj, samples: int):
        """Generate synthetic training data for the target unitary."""
        return random_training_data(unitary, samples)

    def feedforward(
        self,
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Run the full forward pass including an optional quanvolution filter."""
        processed_samples: List[Tuple[qt.Qobj, qt.Qobj]] = []
        for sample, target in samples:
            if self.use_quanv:
                # Apply quanvolution to the computational basis representation
                # of the input state.  The state is assumed to be a pure state
                # on a single qubit per pixel.
                # Convert the state to a 2‑D array of amplitudes.
                data = np.abs(sample.full()).reshape(self.kernel_size, self.kernel_size)
                conv_out = self.quanv.run(data)
                # Embed the scalar conv output into a new qubit register.
                conv_state = qt.tensor(
                    qt.Qobj([[conv_out], [1 - conv_out]]),  # |0> amplitude = 1-conv_out
                    sample,
                )
                processed_samples.append((conv_state, target))
            else:
                processed_samples.append((sample, target))
        return feedforward(self.qnn_arch, unitaries, processed_samples)

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )
