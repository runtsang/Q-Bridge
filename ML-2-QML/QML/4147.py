"""GraphQNNHybrid: quantum‑centric implementation using qutip and qiskit.

This module mirrors the classical GraphQNNHybrid, adding a quantum state‑based
network and a Qiskit expectation head.  The API is identical, enabling
drop‑in replacement for the classical version.  The estimator utilities
support shot‑noise simulation via Qiskit simulators.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import torch
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

Tensor = qt.Qobj
ScalarObservable = Callable[[Tensor], Tensor | float]


# --------------------------------------------------------------------------
#  Helper functions for the quantum network
# --------------------------------------------------------------------------
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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
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


def random_network(qnn_arch: Sequence[int], samples: int):
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

    return list(qnn_arch), unitaries, training_data, target_unitary


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
    qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj
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


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
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


# --------------------------------------------------------------------------
#  Hybrid quantum head using Qiskit
# --------------------------------------------------------------------------
class QuantumCircuitWrapper:
    """
    Thin wrapper around a parameterised Qiskit circuit used as a hybrid head.
    The circuit implements a single‑qubit expectation of Z after a Ry rotation.
    """

    def __init__(self, n_qubits: int, shots: int = 100):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge from PyTorch to the Qiskit hybrid head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid head that forwards activations through a Qiskit circuit."""

    def __init__(self, n_qubits: int, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


# --------------------------------------------------------------------------
#  GraphQNNHybrid quantum version
# --------------------------------------------------------------------------
class GraphQNNHybrid:
    """
    Quantum‑centric graph neural network that mirrors the classical GraphQNNHybrid.
    It supports a hybrid quantum expectation head and fast estimators of
    arbitrary operators with optional shot‑noise.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        *,
        use_hybrid_head: bool = False,
        shift: float = np.pi / 2,
        shots: int = 100,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.arch = list(qnn_arch)
        self.arch, self.unitaries, self.training_data, _ = random_network(self.arch, samples=10)
        self.use_hybrid_head = use_hybrid_head
        self.shift = shift
        self.shots = shots
        if use_hybrid_head:
            self.hybrid = Hybrid(self.arch[-1], shots=shots, shift=shift)
        else:
            self.hybrid = None

        # Parameterised base circuit for estimators
        self.base_circuit = QuantumCircuit(self.arch[-1])
        self.parameters = []
        for i in range(self.arch[-1]):
            p = Parameter(f"theta_{i}")
            self.parameters.append(p)
            self.base_circuit.ry(p, i)

    def forward(self, state: qt.Qobj) -> qt.Qobj:
        """Propagate a single state through all layers."""
        for layer in range(1, len(self.arch)):
            state = _layer_channel(self.arch, self.unitaries, layer, state)
        if self.use_hybrid_head:
            # Use the hybrid head to compute an expectation value
            # For demonstration, we simply evaluate the hybrid head on a dummy input
            dummy_input = torch.tensor([0.0])
            expectation = self.hybrid.forward(dummy_input).item()
            return qt.Qobj(np.array([[expectation]]))
        return state

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------
    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        """Return states at each layer for a batch of samples."""
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Compute adjacency graph from the network's final states."""
        final_states = [activations[-1] for activations in self.feedforward(samples=self._dummy_samples())]
        return fidelity_adjacency(final_states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def _dummy_samples(self) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate dummy samples for graph construction."""
        return [(_random_qubit_state(self.arch[0]), qt.Qobj()) for _ in range(10)]

    # ------------------------------------------------------------------
    #  Estimator utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self.base_circuit.copy()
            mapping = dict(zip(self.parameters, values))
            circ = circ.assign_parameters(mapping, inplace=False)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [rng.normal(complex(val), max(1e-6, 1 / shots)) for val in row]
                noisy.append(noisy_row)
            return noisy
        return results


__all__ = [
    "QuantumCircuitWrapper",
    "HybridFunction",
    "Hybrid",
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
