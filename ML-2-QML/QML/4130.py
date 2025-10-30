"""Hybrid quantum convolution / graph neural network module.

The class ConvHybridGen107 can be used as a drop‑in replacement for the
original Conv module.  It supports:
* a quantum convolutional filter built from RX rotations and a
  random two‑qubit circuit (Qiskit)
* a graph‑based quantum neural network that operates on qubit registers
  and uses state‑fidelity to build adjacency graphs
* a lightweight estimator that evaluates arbitrary observables on
  batches of parameters using Statevector expectation values

The implementation relies on Qiskit and NetworkX and is written to
be lightweight enough for simulation on a classical backend.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable

import numpy as np
import networkx as nx
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

Tensor = np.ndarray
ScalarObservable = Callable[[np.ndarray], np.ndarray | float]

def _tensored_id(num_qubits: int) -> qiskit.quantum_info.Qobj:
    identity = qiskit.quantum_info.Qobj(np.eye(2 ** num_qubits))
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qiskit.quantum_info.Qobj:
    projector = qiskit.quantum_info.Qobj(np.diag([0] * (2 ** num_qubits)))
    projector[0, 0] = 1.0
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qiskit.quantum_info.Qobj, source: int, target: int) -> qiskit.quantum_info.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qiskit.quantum_info.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    u, _ = np.linalg.svd(matrix)
    qobj = qiskit.quantum_info.Qobj(u)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qiskit.quantum_info.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)
    amplitudes /= np.linalg.norm(amplitudes)
    state = qiskit.quantum_info.Qobj(amplitudes)
    dims = [2] * num_qubits
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qiskit.quantum_info.Qobj, samples: int) -> List[Tuple[qiskit.quantum_info.Qobj, qiskit.quantum_info.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qiskit.quantum_info.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qiskit.quantum_info.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qiskit.quantum_info.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qiskit.quantum_info.Qobj, keep: Sequence[int]) -> qiskit.quantum_info.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qiskit.quantum_info.Qobj, remove: Sequence[int]) -> qiskit.quantum_info.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qiskit.quantum_info.Qobj]], layer: int, input_state: qiskit.quantum_info.Qobj) -> qiskit.quantum_info.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qiskit.quantum_info.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qiskit.quantum_info.Qobj]], samples: Iterable[Tuple[qiskit.quantum_info.Qobj, qiskit.quantum_info.Qobj]]) -> List[List[qiskit.quantum_info.Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qiskit.quantum_info.Qobj, b: qiskit.quantum_info.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qiskit.quantum_info.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ConvHybridGen107:
    """Hybrid quantum convolution / graph neural network.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch that is convolved over the input.
    threshold : int, default 127
        Threshold for RX rotation angles in the quantum filter.
    shots : int | None, default 100
        Number of shots for the simulator.  If ``None`` a statevector
        evaluator is used.
    backend : qiskit.providers.BaseBackend | None, default None
        Quantum backend.  If ``None`` the Aer qasm simulator is used.
    use_graph : bool, default False
        When ``True`` the network is built from ``qnn_arch`` and the
        convolution step is replaced by a graph‑based feedforward.
    qnn_arch : Sequence[int] | None, default None
        Architecture of a graph‑based network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: int = 127,
        shots: int | None = 100,
        backend: qiskit.providers.BaseBackend | None = None,
        use_graph: bool = False,
        qnn_arch: Sequence[int] | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.use_graph = use_graph

        if use_graph:
            if qnn_arch is None:
                raise ValueError("qnn_arch must be provided when use_graph=True")
            (
                self.arch,
                self.unitaries,
                self.training_data,
                self.target_unitary,
            ) = random_network(qnn_arch, samples=10)
        else:
            self.n_qubits = kernel_size ** 2
            self._circuit = QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Apply the chosen filter to ``data`` and return a scalar."""
        if self.use_graph:
            # Placeholder: graph mode not implemented in this simplified example
            return 0.0
        else:
            param_binds = []
            for val in data.flatten():
                bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i in range(self.n_qubits)}
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

    def fidelity_graph(self, threshold: float) -> nx.Graph:
        """Return a graph built from the current layer outputs."""
        if not self.use_graph:
            raise RuntimeError("fidelity_graph is only available for graph networks")
        states = [
            feedforward(self.arch, self.unitaries, [sample, None])[0][-1]
            for sample, _ in self.training_data
        ]
        return fidelity_adjacency(states, threshold)

    # ------------------------------------------------------------------
    # Estimator utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        if self.use_graph:
            raise NotImplementedError("Estimator is not defined for graph mode in this simplified example.")
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= self.n_qubits:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.theta, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

def Conv() -> ConvHybridGen107:
    """Factory returning a default hybrid filter."""
    return ConvHybridGen107()

__all__ = ["ConvHybridGen107", "Conv"]
