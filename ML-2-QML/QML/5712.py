"""Unified estimator that evaluates quantum circuits and can generate
graph‑based quantum neural networks.

The module contains:
    * UnifiedGraphEstimator – evaluates a Qiskit QuantumCircuit with
      optional shot noise.
    * random_network – generate a random QNN architecture with unitary
      layers (mirrors the GraphQNN random_network).
    * feedforward – propagate a sample state through the QNN layers.
    * fidelity_adjacency – build a graph from state fidelities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any, Optional
import itertools
import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import qutip as qt
import scipy as sc
import torch

# ----------------------------------------------------------------------
# Utility functions – random linear, training data, etc. (classical)
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

# ----------------------------------------------------------------------
# Quantum neural network helpers
# ----------------------------------------------------------------------
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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
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

# ----------------------------------------------------------------------
# Unified estimator – quantum backend
# ----------------------------------------------------------------------
class UnifiedGraphEstimator:
    """Estimator that evaluates a Qiskit QuantumCircuit with optional shot noise."""

    def __init__(self, circuit: QuantumCircuit, *, shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _evaluate(self, observables: Iterable[Any],
                  parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values of the circuit."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(
                dict(zip(self.circuit.parameters, params)), inplace=False
            )
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(self, observables: Iterable[Any],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        raw = self._evaluate(observables, parameter_sets)
        if self.shots is None:
            return [[float(v) for v in row] for row in raw]
        # Shot noise: sample from normal distribution around expectation
        noisy = []
        for row in raw:
            noisy_row = [float(self._rng.normal(float(v), max(1e-6, 1 / self.shots))) for v in row]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)
