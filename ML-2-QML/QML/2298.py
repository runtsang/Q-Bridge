"""Graph‑based hybrid neural network – quantum implementation.

The quantum branch mirrors the classical logic but operates on
Qobj states and unitaries.  It also provides a FastBaseEstimator
that evaluates expectation values via a Statevector simulator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List

import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# Helper utilities
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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[tuple[qt.Qobj, qt.Qobj]]:
    """Generate input–output pairs for a target unitary."""
    dataset: List[tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random quantum network and training set."""
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
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Propagate each sample through the quantum network."""
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
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
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
# Quantum estimator
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Compute expectation values for a parametrised circuit."""

    def __init__(self, circuit: qt.Qobj):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> qt.Qobj:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = qt.Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Hybrid GraphQNN class (quantum side)
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """Graph‑based quantum neural network with a unified API.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [3, 5, 2].
    samples : int, optional
        Number of synthetic training samples to generate.
    """

    def __init__(self, arch: Sequence[int], samples: int = 1000) -> None:
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(
            arch, samples
        )

    def feedforward(self, inputs: qt.Qobj) -> List[qt.Qobj]:
        """Return activations for a single input state."""
        layerwise = [inputs]
        current = inputs
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.unitaries, layer, current)
            layerwise.append(current)
        return layerwise

    def fidelity_graph(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Graph of fidelities between last‑layer outputs."""
        last_outputs = [act[-1] for act in feedforward(self.arch, self.unitaries, self.training_data)]
        return fidelity_adjacency(last_outputs, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: Iterable[qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Batch evaluation using the quantum FastBaseEstimator."""
        estimator = FastBaseEstimator(self.target_unitary)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
    "FastBaseEstimator",
    "GraphQNNHybrid",
]
