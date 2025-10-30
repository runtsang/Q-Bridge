"""Hybrid estimator for quantum circuits.

This module implements a `HybridEstimator` that can wrap any Qiskit
`QuantumCircuit`.  It provides a unified `evaluate` API that accepts
an iterable of `BaseOperator` observables and a list of parameter sets.
Optional shot‑noise can be simulated by adding Gaussian noise to the
expectation values.

Helper utilities for generating random quantum networks and building
quantum kernels, samplers, and quanvolution filters are included,
mirroring the `GraphQNN` and `SamplerQNN` references.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import qutip as qt
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
#  Random unitary and network generation (Quantum version)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return a tensor‑product identity."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Return a tensor‑product zero projector."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap two registers in a Qobj."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Return a random unitary matrix as a Qobj."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Return a random pure state as a Qobj."""
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training data by applying a unitary to random states."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(
    arch: Sequence[int], samples: int = 100
) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random quantum feed‑forward network.

    Returns:
        arch: The architecture list.
        unitaries: List of lists of Qobj representing each layer.
        training_data: List of (input_state, target_state) tuples.
        target_unitary: The final unitary (ground truth).
    """
    target_unitary = _random_qubit_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(arch)):
        num_inputs = arch[layer - 1]
        num_outputs = arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(arch), unitaries, training_data, target_unitary


# --------------------------------------------------------------------------- #
#  Fidelity‑based graph construction
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= threshold receive weight 1.0.
    If ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` receive ``secondary_weight``.
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


# --------------------------------------------------------------------------- #
#  Quantum sampler and kernel circuits
# --------------------------------------------------------------------------- #
def create_sampler_qnn() -> QuantumCircuit:
    """Return a simple parameterized sampler circuit (SamplerQNN)."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc


def create_quantum_kernel(num_qubits: int, input_dim: int, weight_dim: int) -> QuantumCircuit:
    """Build a simple quantum kernel circuit.

    The circuit encodes the input vector into rotation angles and applies
    a small variational layer.  It can be used as a feature map for
    hybrid models.
    """
    inputs = ParameterVector("input", input_dim)
    weights = ParameterVector("weight", weight_dim)

    qc = QuantumCircuit(num_qubits)
    for i, wire in enumerate(range(num_qubits)):
        qc.ry(inputs[i], wire)
    for i, wire in enumerate(range(num_qubits)):
        qc.ry(weights[i], wire)
    return qc


def create_quanvolution_filter() -> QuantumCircuit:
    """Return a quantum circuit that implements a 2×2 patch quanvolution."""
    # For simplicity, use a 4‑qubit circuit that applies a random layer
    # to each 2×2 patch.  The circuit is a placeholder and can be
    # replaced by a more sophisticated kernel.
    qc = QuantumCircuit(4)
    qc.h([0, 1, 2, 3])
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.barrier()
    qc.ry(0.1, 0)
    qc.ry(0.2, 1)
    qc.ry(0.3, 2)
    qc.ry(0.4, 3)
    return qc


# --------------------------------------------------------------------------- #
#  HybridEstimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """Estimator that wraps a Qiskit `QuantumCircuit`.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized quantum circuit that will be evaluated for each
        parameter set.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of `BaseOperator` objects to measure.
        parameter_sets
            Sequence of parameter vectors (each a sequence of floats).
        shots
            If provided, Gaussian noise with stddev 1/sqrt(shots) is added
            to each mean value to mimic sampling noise.
        seed
            Random seed for the noise generator.

        Returns
        -------
        List[List[complex]]
            Nested list where each inner list contains the expectation
            values for a single parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                    for val in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = [
    "HybridEstimator",
    "random_network",
    "fidelity_adjacency",
    "create_sampler_qnn",
    "create_quantum_kernel",
    "create_quanvolution_filter",
]
