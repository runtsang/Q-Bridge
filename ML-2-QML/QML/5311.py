"""GraphQNNGen197: quantum counterpart with optional auto‑encoding and sampler circuits."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Tuple as TupleType

import networkx as nx
import qutip as qt
import scipy as sc
import qiskit
import numpy as np

# Quantum helpers
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector

# Auxiliary helpers
from Autoencoder import Autoencoder  # returns a SamplerQNN instance
from FCL import FCL  # quantum‑style fully‑connected layer

Tensor = qt.Qobj

# -------------------------------------------------------------------------
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
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random chain of unitary layers and training data."""
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
):
    """Forward pass through a quantum feed‑forward network."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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


# -------------------------------------------------------------------------
class QuantumFCL:
    """Simple parameterised quantum circuit for a fully‑connected layer."""

    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


# -------------------------------------------------------------------------
class GraphQNNGen197:
    """Quantum‑aware counterpart of :class:`GraphQNNGen197`."""

    def __init__(
        self,
        arch: Sequence[int],
        *,
        autoencoder: bool = False,
        sampler: bool = False,
    ) -> None:
        self.arch = list(arch)
        self.autoencoder = autoencoder
        self.sampler = sampler

        # Storage for generated components
        self.unitaries: List[List[qt.Qobj]] | None = None
        self.target_unitary: qt.Qobj | None = None
        self.autoencoder_qnn: SamplerQNN | None = None
        self.sampler_qnn: SamplerQNN | None = None
        self.fcl: QuantumFCL | None = None

        if self.autoencoder:
            self.autoencoder_qnn = Autoencoder()  # returns a SamplerQNN
        if self.sampler:
            self.sampler_qnn = SamplerQNN()
        # Example backend for the FCL
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.fcl = QuantumFCL(1, backend, shots=100)

    # -------------------------------------------------------------------------
    def random_network(self, samples: int):
        """Generate quantum network and training data."""
        arch, unitaries, training_data, target_unitary = random_network(self.arch, samples)
        self.unitaries = unitaries
        self.target_unitary = target_unitary
        return arch, unitaries, training_data, target_unitary

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
        """Run samples through the quantum network."""
        if self.unitaries is None:
            raise RuntimeError("Unitary chain not initialised; call random_network() first.")
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        """Return a graph built from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # -------------------------------------------------------------------------
    def autoencoder_qnn(self) -> SamplerQNN:
        """Return the quantum auto‑encoder sampler."""
        if not self.autoencoder or self.autoencoder_qnn is None:
            raise RuntimeError("Autoencoder not configured.")
        return self.autoencoder_qnn

    def sampler_qnn(self) -> SamplerQNN:
        """Return the quantum sampler."""
        if not self.sampler or self.sampler_qnn is None:
            raise RuntimeError("Sampler not configured.")
        return self.sampler_qnn

    def fcl_run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the fully‑connected quantum circuit."""
        if self.fcl is None:
            raise RuntimeError("FCL not configured.")
        return self.fcl.run(thetas)
