"""Hybrid quantum estimator that blends parameterized quantum circuits with graph‑based fidelity analysis.

The class builds a layer‑wise quantum neural network using Qiskit,
estimates expectation values via StatevectorEstimator, and can
construct a weighted graph of quantum states based on fidelity.
A ``run`` method evaluates the circuit for a supplied sequence of angles,
returning the expectation of the Y observable on the output qubit.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import qutip as qt
import scipy as sc

__all__ = ["HybridEstimatorQNN"]


# ----------------- Quantum utilities (adapted from GraphQNN.py) -----------------
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


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
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


def _state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------- Hybrid quantum estimator ------------------------------------
class HybridEstimatorQNN:
    """
    Quantum hybrid estimator.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[1, 2, 1]``.
    use_graph : bool, default=True
        Whether to expose a ``graph`` method that constructs a fidelity graph
        of intermediate quantum states.
    graph_threshold : float, default=0.9
        Threshold for edge creation in the fidelity graph.
    """

    def __init__(
        self,
        arch: Sequence[int],
        use_graph: bool = True,
        graph_threshold: float = 0.9,
    ) -> None:
        self.arch = list(arch)
        self.use_graph = use_graph
        self.graph_threshold = graph_threshold

        # Build parameterised circuit
        self.input_params: List[Parameter] = []
        self.weight_params: List[Parameter] = []

        qc = QuantumCircuit(max(self.arch))
        for qubit in range(max(self.arch)):
            self.input_params.append(Parameter(f"x{qubit}"))
            self.weight_params.append(Parameter(f"w{qubit}"))
            qc.h(qubit)
            qc.ry(self.input_params[qubit], qubit)
            qc.rx(self.weight_params[qubit], qubit)

        # Observable: Pauli Y on the last qubit of the final layer
        obs = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

        # Estimator
        estimator = StatevectorEstimator()

        self.estimator_qnn = QEstimatorQNN(
            circuit=qc,
            observables=obs,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

        # Prepare quantum graph machinery
        self.arch_q, self.unitaries, self.training_data, self.target_unitary = random_network(self.arch, samples=10)

    # ---------------------------------------------------------------------------

    def evaluate(self, inputs: Iterable[float]) -> np.ndarray:
        """Run the quantum circuit for a sequence of input angles."""
        input_dict = {p: val for p, val in zip(self.input_params, inputs)}
        result = self.estimator_qnn.evaluate(input_dict)
        return np.array(result)

    # ---------------------------------------------------------------------------

    def predict(self, inputs: Iterable[float]) -> np.ndarray:
        """Alias for evaluate."""
        return self.evaluate(inputs)

    # ---------------------------------------------------------------------------

    def graph(self, samples: int = 100, threshold: float | None = None) -> nx.Graph:
        """Return a weighted graph of intermediate quantum states constructed from random samples."""
        if not self.use_graph:
            raise RuntimeError("Graph construction disabled in this instance.")
        threshold = threshold if threshold is not None else self.graph_threshold

        # Random samples from training data
        random_samples = random_training_data(self.target_unitary, samples)
        states = [s for s, _ in random_samples]
        return fidelity_adjacency(states, threshold)

    # ---------------------------------------------------------------------------

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic a quantum fully‑connected layer.

        Thetas are interpreted as input angles for the first qubit,
        the circuit is executed, and the expectation of Y on the last qubit
        is returned as a single‑element array.
        """
        return self.evaluate(thetas)

    # ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arch={self.arch})"
