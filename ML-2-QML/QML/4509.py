"""Quantum‑centric hybrid estimator that mirrors the classical API.

The module implements the same public API but uses Qiskit for circuit
evaluation and QuTiP for graph‑based QNN propagation.  The class name
is identical to the classical version so that a single interface can
be used in experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ----------------------------------------------------------------------
# Qiskit estimator
# ----------------------------------------------------------------------
class _QiskitEstimator:
    """Fast deterministic estimator for a parametrised Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class _NoisyQiskitEstimator(_QiskitEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(val.real, max(1e-6, 1 / shots)),
                    rng.normal(val.imag, max(1e-6, 1 / shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# QuTiP graph‑QNN utilities
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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amps = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amps /= np.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        st = _random_qubit_state(n)
        dataset.append((st, unitary * st))
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
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
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
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------------------------------------------------------------
# Quantum fully‑connected layer
# ----------------------------------------------------------------------
class QuantumCircuit:
    """Parameterised two‑qubit circuit used as a stand‑in for a quantum layer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
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
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


def FCL() -> QuantumCircuit:
    """Return a one‑qubit quantum circuit mimicking a fully‑connected layer."""
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return QuantumCircuit(1, backend, shots=100)


# ----------------------------------------------------------------------
# Hybrid quantum‑classical classifier
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and a Qiskit circuit."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuit,
        shift: float,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads: List[float] = []
        for idx, val in enumerate(inputs.numpy()):
            exp_r = ctx.circuit.run([val + shift[idx]])
            exp_l = ctx.circuit.run([val - shift[idx]])
            grads.append(exp_r - exp_l)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a Qiskit circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.circuit, self.shift)


class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)


# ----------------------------------------------------------------------
# Unified estimator
# ----------------------------------------------------------------------
class HybridEstimator:
    """Quantum‑centric estimator mirroring the classical API.

    Parameters
    ----------
    model : Union[QuantumCircuit, Tuple, nn.Module]
        * ``QuantumCircuit`` → Qiskit estimator.
        * ``Tuple`` (arch, unitaries, data, target) → graph‑QNN estimator.
        * ``nn.Module`` → hybrid classifier.
    model_type : str, optional
        One of ``'qiskit'``, ``'graph'`` or ``'hybrid'``.  The default is inferred.
    """

    def __init__(self, model, model_type: str | None = None) -> None:
        if model_type is None:
            if isinstance(model, QuantumCircuit):
                model_type = "qiskit"
            elif isinstance(model, tuple) and len(model) == 4:
                model_type = "graph"
            elif isinstance(model, nn.Module):
                model_type = "hybrid"
            else:
                raise TypeError("Could not infer model_type from the provided model.")
        self.model_type = model_type
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Dispatch evaluation to the underlying quantum backend."""
        if self.model_type == "qiskit":
            estimator = _NoisyQiskitEstimator(self.model)
            return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
        elif self.model_type == "graph":
            return self._evaluate_graph(observables, parameter_sets)
        elif self.model_type == "hybrid":
            return self._evaluate_hybrid(observables, parameter_sets)
        else:
            raise ValueError(f"Unsupported model_type {self.model_type!r}")

    # ------------------------------------------------------------------
    # Backend specific helpers
    # ------------------------------------------------------------------
    def _evaluate_graph(
        self,
        observables: Iterable[Callable[[qt.Qobj], qt.Qobj | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        arch, unitaries, data, _ = self.model
        results: List[List[complex]] = []
        for idx in parameter_sets:
            sample, _ = data[idx]
            activations = feedforward(arch, unitaries, [(sample, _)])
            row = [float(ob(activations[-1])) for ob in observables]
            results.append(row)
        return results

    def _evaluate_hybrid(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            out = self.model.forward(torch.tensor(params))
            row = [float(ob(out)) for ob in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_qiskit(cls, circuit: QuantumCircuit) -> "HybridEstimator":
        return cls(circuit, "qiskit")

    @classmethod
    def from_graph(cls, arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], data: Sequence[Tuple[qt.Qobj, qt.Qobj]]) -> "HybridEstimator":
        return cls((arch, unitaries, data, None), "graph")

    @classmethod
    def from_hybrid(cls, module: nn.Module) -> "HybridEstimator":
        return cls(module, "hybrid")


__all__ = [
    "HybridEstimator",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
    "QuantumCircuit",
    "FCL",
    "QCNet",
    "Hybrid",
    "HybridFunction",
]
