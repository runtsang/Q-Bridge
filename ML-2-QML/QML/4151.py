"""Quantum‑classical hybrid graph neural network module.

Provides a class `GraphQNNHybrid` that mirrors the classical
implementation but uses qutip unitaries, a quantum quanvolution
filter and a Qiskit EstimatorQNN for regression.  All helper
functions from the original QML seed are preserved and extended.
"""

from __future__ import annotations

import itertools
import numpy as np
import qutip as qt
import scipy as sc
import networkx as nx
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

# ---- Utility functions -------------------------------------------------------

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


def random_training_data(unitary: qt.Qobj, samples: int):
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
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


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
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
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---- Quantum convolution filter ----------------------------------------------

class QuanvCircuit:
    """Quantum filter that emulates a 2‑D quanvolution layer."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        total = sum(result.values())
        ones = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return ones / (self.shots * self.n_qubits)

# ---- Quantum estimator wrapper -----------------------------------------------

class QuantumEstimator:
    """Wrap Qiskit EstimatorQNN to produce a regression output."""
    def __init__(self):
        params = [Parameter("x"), Parameter("w")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        obs = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.model = QiskitEstimatorQNN(circuit=qc,
                                        observables=obs,
                                        input_params=[params[0]],
                                        weight_params=[params[1]],
                                        estimator=estimator)

    def __call__(self, x: float, w: float) -> float:
        return self.model(x, w)

# ---- Hybrid quantum graph neural network -------------------------------------

class GraphQNNHybrid:
    """
    Quantum‑classical hybrid GNN that uses qutip unitaries for propagation,
    a quantum quanvolution filter, and a Qiskit EstimatorQNN for regression.
    """
    def __init__(self,
                 arch: Sequence[int],
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 use_fidelity: bool = True) -> None:
        self.arch = list(arch)
        self.use_fidelity = use_fidelity
        self.conv = QuanvCircuit(conv_kernel,
                                 Aer.get_backend("qasm_simulator"),
                                 shots=100,
                                 threshold=conv_threshold)
        self.estimator = QuantumEstimator()

        self.units: list[list[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            in_n = arch[layer - 1]
            out_n = arch[layer]
            ops: list[qt.Qobj] = []
            for _ in range(out_n):
                op = _random_qubit_unitary(in_n + 1)
                if out_n > 1:
                    op = qt.tensor(_random_qubit_unitary(in_n + 1), _tensored_id(out_n - 1))
                    op = _swap_registers(op, in_n, in_n + _)
                ops.append(op)
            self.units.append(ops)

    def feedforward(self, state: qt.Qobj) -> list[qt.Qobj]:
        """Propagate a state through the network."""
        states = [state]
        for layer in range(1, len(self.arch)):
            state = _layer_channel(self.arch, self.units, layer, state)
            states.append(state)
        return states

    def build_adjacency(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        if not self.use_fidelity:
            return nx.Graph()
        return fidelity_adjacency(states, threshold)

    def random_network(self, samples: int):
        """Return random network and training data."""
        return random_network(self.arch, samples)

    def run_quantum_conv(self, data: np.ndarray) -> float:
        """Apply the quantum convolution filter to 2‑D data."""
        return self.conv.run(data)

    def estimate(self, x: float, w: float) -> float:
        """Predict using the quantum estimator."""
        return self.estimator(x, w)

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "QuanvCircuit",
    "QuantumEstimator",
]
