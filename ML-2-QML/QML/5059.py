"""Quantum implementation of GraphQNN with hybrid estimator support.

This module mirrors the classical GraphQNN utilities but replaces tensor
operations with qiskit primitives.  It also provides a FastBaseEstimator
for parameterised circuits and a simple quantum regression network.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Callable

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator
import torch
from torch import nn

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Classical utilities – graph neural network
# --------------------------------------------------------------------------- #

def classical_random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def classical_random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def classical_random_network(qnn_arch: Sequence[int], samples: int):
    weights = [
        classical_random_linear(in_f, out_f)
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = classical_random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def classical_feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def classical_state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm) ** 2)


def classical_fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = classical_state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Quantum utilities – graph neural network
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return q


def quantum_random_training_data(unitary: np.ndarray, samples: int):
    dataset = []
    dim = unitary.shape[0]
    for _ in range(samples):
        vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        vec /= np.linalg.norm(vec)
        state = Statevector(vec)
        target = state.evolve(unitary)
        dataset.append((state, target))
    return dataset


def quantum_random_network(qnn_arch: List[int], samples: int):
    layer_units: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        ops = []
        for _ in range(out_f):
            ops.append(_random_qubit_unitary(in_f + 1))
        layer_units.append(ops)

    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = quantum_random_training_data(target_unitary, samples)
    return list(qnn_arch), layer_units, training_data, target_unitary


def _layer_channel(
    qnn_arch: List[int],
    layer_units: List[List[np.ndarray]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    op = layer_units[layer][0]
    return input_state.evolve(op)


def quantum_feedforward(
    qnn_arch: List[int],
    layer_units: List[List[np.ndarray]],
    samples: Iterable[tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored: List[List[Statevector]] = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, layer_units, layer, current)
            states.append(current)
        stored.append(states)
    return stored


def quantum_state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs(np.vdot(a.data, b.data)) ** 2


def quantum_fidelity_adjacency(
    states: List[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = quantum_state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Estimators – quantum
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
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
                rng.normal(val.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Quantum regression network – quantum
# --------------------------------------------------------------------------- #

def EstimatorQNN():
    """Return a tiny parameterised quantum circuit used as a regression model."""
    input_params = [Parameter(f"x{i}") for i in range(2)]
    weight_params = [Parameter(f"w{i}") for i in range(2)]

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(input_params[0], 0)
    qc.rx(weight_params[0], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])

    return qc, observable, input_params, weight_params


# --------------------------------------------------------------------------- #
#  Quantum convolution filter – quanvolution
# --------------------------------------------------------------------------- #

def Conv():
    """Return a quantum filter that mimics the classical Conv filter."""
    class QuanvCircuit:
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            self.n_qubits = kernel_size ** 2
            self._circuit = QuantumCircuit(self.n_qubits)
            self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = AerSimulator()
            self.shots = 100
            self.threshold = threshold

        def run(self, data):
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = self.backend.run(
                self._circuit,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self._circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    return QuanvCircuit()


# --------------------------------------------------------------------------- #
#  Hybrid GraphQNN – unified interface
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """
    Quantum‑classical hybrid GraphQNN that exposes the same API as the
    classical implementation.  In quantum mode the forward pass is
    delegated to a parameterised circuit, while the classical mode
    uses a small torch network.
    """

    def __init__(
        self,
        arch: List[int],
        *,
        use_quantum: bool = True,
        shots: int | None = 100,
        threshold: float = 0.5,
    ) -> None:
        self.arch = list(arch)
        self.use_quantum = use_quantum
        self.threshold = threshold
        self.shots = shots

        if use_quantum:
            self.circuit, self.observable, self.input_params, self.weight_params = EstimatorQNN()
            self.estimator = FastEstimator(self.circuit)
        else:
            # Classical fallback – simple torch network
            class TorchEstimator(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(2, 8),
                        nn.Tanh(),
                        nn.Linear(8, 4),
                        nn.Tanh(),
                        nn.Linear(4, 1),
                    )

                def forward(self, inputs: Tensor) -> Tensor:
                    return self.net(inputs)

            class TorchEstimatorBase:
                def __init__(self, model: nn.Module) -> None:
                    self.model = model

                def evaluate(
                    self,
                    observables: Iterable[Callable[[Tensor], Tensor | float]],
                    parameter_sets: List[List[float]],
                ) -> List[List[float]]:
                    observables = list(observables) or [lambda x: x.mean(dim=-1)]
                    results: List[List[float]] = []
                    self.model.eval()
                    with torch.no_grad():
                        for params in parameter_sets:
                            inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                            outputs = self.model(inputs)
                            row: List[float] = []
                            for obs in observables:
                                val = obs(outputs)
                                row.append(float(val.mean().cpu()))
                            results.append(row)
                    return results

            class TorchEstimatorNoise(TorchEstimatorBase):
                def evaluate(
                    self,
                    observables: Iterable[Callable[[Tensor], Tensor | float]],
                    parameter_sets: List[List[float]],
                    *,
                    shots: int | None = None,
                    seed: int | None = None,
                ) -> List[List[float]]:
                    raw = super().evaluate(observables, parameter_sets)
                    if shots is None:
                        return raw
                    rng = np.random.default_rng(seed)
                    noisy: List[List[float]] = []
                    for row in raw:
                        noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                        noisy.append(noisy_row)
                    return noisy

            self.estimator = TorchEstimatorNoise(TorchEstimator())
            # initialise classical weights for feedforward
            self.weights = [
                classical_random_linear(in_f, out_f)
                for in_f, out_f in zip(arch[:-1], arch[1:])
            ]

    def feedforward(
        self, samples: List[tuple[Statevector, Statevector]]
    ) -> List[List[Statevector]]:
        if self.use_quantum:
            states: List[List[Statevector]] = []
            for sample, _ in samples:
                bound = self.circuit.assign_parameters(
                    dict(zip(self.input_params, sample.data)), inplace=False
                )
                state = Statevector.from_instruction(bound)
                states.append([state])
            return states
        else:
            # Use classical feedforward with torch weights
            return classical_feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        states: List[Statevector],
        threshold: float | None = None,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        if threshold is None:
            threshold = self.threshold
        return quantum_fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    def random_training_data(self, samples: int) -> List[tuple[Statevector, Statevector]]:
        if self.use_quantum:
            target_unitary = Statevector.from_instruction(self.circuit).to_matrix()
            return quantum_random_training_data(target_unitary, samples)
        else:
            dummy_weight = torch.eye(2)
            return classical_random_training_data(dummy_weight, samples)

    def conv_filter(self, data) -> float:
        filter_obj = Conv()
        return filter_obj.run(data)

    def __repr__(self) -> str:
        mode = "Quantum" if self.use_quantum else "Classical"
        return f"<GraphQNNHybrid ({mode}) arch={self.arch}>"

__all__ = [
    "GraphQNNHybrid",
    "classical_feedforward",
    "classical_fidelity_adjacency",
    "classical_random_network",
    "classical_random_training_data",
    "classical_state_fidelity",
    "FastBaseEstimator",
    "FastEstimator",
    "EstimatorQNN",
    "Conv",
]
