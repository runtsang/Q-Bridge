"""GraphQNNGen452 – quantum implementation.

This module mirrors the classical GraphQNNGen452 but operates on
parameterised Qiskit circuits and state‑vector simulation.  The
public API is identical, enabling seamless substitution in a hybrid
pipeline.

The design incorporates ideas from the seed projects:

* Random unitary generation and synthetic training data (GraphQNN).
* Fraud‑detection style photonic layers are emulated with
  U3 rotations, RZ, RX, and two‑qubit CNOT entanglers.
* A simple Qiskit EstimatorQNN backbone for regression.
* Fast evaluation with optional Gaussian shot noise (FastBaseEstimator).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

Tensor = Statevector
ScalarObservable = Callable[[Tensor], complex | float]

# Alias for API compatibility
EstimatorQNN = QiskitEstimatorQNN


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Return a 2‑qubit circuit that mimics the photonic stack."""
    qc = QuantumCircuit(2)

    def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, clip: bool):
        # Two‑qubit entangler (CNOT)
        qc.cx(0, 1)
        # Single‑qubit rotations
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        # Squeezing → simulate with RX and RZ
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            angle = r if not clip else _clip(r, 5)
            qc.rx(angle, i)
            qc.rz(phi, i)
        # Displacement → RX
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            angle = r if not clip else _clip(r, 5)
            qc.rx(angle, i)
            qc.rz(phi, i)
        # Kerr → RZ
        for i, k in enumerate(params.kerr):
            angle = k if not clip else _clip(k, 1)
            qc.rz(angle, i)
        # Entangler again
        qc.cx(0, 1)

    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


def _random_qubit_unitary(num_qubits: int) -> Operator:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return Operator(q)


def _random_qubit_state(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random target unitary and synthetic training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = [
        (_random_qubit_state(qnn_arch[-1]), target_unitary @ _random_qubit_state(qnn_arch[-1]))
        for _ in range(samples)
    ]

    circuits: List[QuantumCircuit] = []
    for _ in range(len(qnn_arch) - 1):
        qc = QuantumCircuit(qnn_arch[-1])
        qc.append(_random_qubit_unitary(qnn_arch[-1]).to_instruction(), qc.qubits)
        circuits.append(qc)

    return qnn_arch, circuits, training_data, target_unitary


def _layer_channel(circuit: QuantumCircuit, input_state: Statevector) -> Statevector:
    return input_state.evolve(circuit)


def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[QuantumCircuit],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored_states = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for qc in circuits:
            current = _layer_channel(qc, current)
            states.append(current)
        stored_states.append(states)
    return stored_states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
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


class FastBaseEstimator:
    """Evaluate expectation values for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


def random_fraud_params(num_layers: int, seed: int | None = None) -> Tuple[FraudLayerParameters, List[FraudLayerParameters]]:
    rng = random.Random(seed)
    input_params = FraudLayerParameters(
        bs_theta=rng.uniform(-np.pi, np.pi),
        bs_phi=rng.uniform(-np.pi, np.pi),
        phases=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        squeeze_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
        squeeze_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        displacement_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
        displacement_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
        kerr=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
    )
    layers = [
        FraudLayerParameters(
            bs_theta=rng.uniform(-np.pi, np.pi),
            bs_phi=rng.uniform(-np.pi, np.pi),
            phases=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            squeeze_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
            squeeze_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            displacement_r=(rng.uniform(0, 1), rng.uniform(0, 1)),
            displacement_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
            kerr=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
        )
        for _ in range(num_layers)
    ]
    return input_params, layers


class GraphQNNGen452:
    """
    Unified quantum graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer.
    use_fraud : bool, default False
        If True, the network is built from fraud‑detection style layers.
    use_estimator : bool, default False
        If True, the network is the Qiskit EstimatorQNN backbone.
    seed : int | None, default None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        use_fraud: bool = False,
        use_estimator: bool = False,
        seed: int | None = None,
    ) -> None:
        self.arch = list(arch)
        self.seed = seed
        self.circuits: List[QuantumCircuit] = []

        if use_estimator:
            # Simple single‑qubit EstimatorQNN circuit
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.ry(0.5, 0)
            qc.rx(0.3, 0)
            self.circuit = qc
        elif use_fraud:
            input_params, layers = random_fraud_params(len(arch) - 1, seed)
            self.circuit = build_fraud_detection_circuit(input_params, layers)
        else:
            # Default random circuit
            self.circuit = QuantumCircuit(arch[0])
            for _ in range(len(arch) - 1):
                self.circuit.append(_random_qubit_unitary(arch[0]).to_instruction(), self.circuit.qubits)
            self.circuits = [self.circuit]

    # --------------------------------------------------------------------- #
    # 5.1  Random network generation
    # --------------------------------------------------------------------- #
    def random_network(self, samples: int) -> Tuple[Sequence[int], List[QuantumCircuit], List[Tuple[Statevector, Statevector]], Statevector]:
        arch, circuits, training_data, target_unitary = random_network(self.arch, samples)
        self.circuits = circuits
        return arch, circuits, training_data, target_unitary

    # --------------------------------------------------------------------- #
    # 5.2  Feed‑forward
    # --------------------------------------------------------------------- #
    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        if not self.circuits:
            raise RuntimeError("No circuit stack generated. Call random_network first.")
        return feedforward(self.arch, self.circuits, samples)

    # --------------------------------------------------------------------- #
    # 5.3  Fidelity helpers
    # --------------------------------------------------------------------- #
    def state_fidelity(self, a: Statevector, b: Statevector) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # --------------------------------------------------------------------- #
    # 5.4  Estimator evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "GraphQNNGen452",
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "EstimatorQNN",
    "FastBaseEstimator",
]
