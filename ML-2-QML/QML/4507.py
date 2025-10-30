"""Hybrid FastBaseEstimator for quantum Qiskit models.

Provides deterministic evaluation via state‑vector simulation and shot‑based sampling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector, BaseOperator
from qiskit.primitives import StatevectorSampler, Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info.operators.base_operator import SparsePauliOp

# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Deterministic and shot‑based evaluation of a QuantumCircuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Deterministic expectation values via state‑vector simulation."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Shot‑based evaluation using a StatevectorSampler."""
        if shots is None:
            return self.evaluate(observables, parameter_sets)

        sampler = StatevectorSampler(seed=seed)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            samples = sampler.run(bound_circ, shots=shots)
            row = [samples.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum model factories
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int, depth: int
) -> tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """Construct a layered variational ansatz with explicit encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Quantum Autoencoder
# --------------------------------------------------------------------------- #

def Autoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a quantum auto‑encoder implemented with a SamplerQNN."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumCircuit(num_latent + 2 * num_trash + 1)
        circuit = ansatz(num_latent + num_trash)
        qr.compose(circuit, range(0, num_latent + num_trash), inplace=True)
        auxiliary = num_latent + 2 * num_trash
        qr.h(auxiliary)
        for i in range(num_trash):
            qr.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        qr.h(auxiliary)
        qr.measure(auxiliary, 0)
        return qr

    circuit = auto_encoder_circuit(num_latent, num_trash)

    def identity_interpret(x: list[float]) -> list[float]:
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Quantum QCNN
# --------------------------------------------------------------------------- #

def QCNN() -> EstimatorQNN:
    """Return a QCNN‑style quantum neural network built with EstimatorQNN."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = conv_circuit(params[param_index : param_index + 3])
            qc.append(sub, [q1, q2])
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = conv_circuit(params[param_index : param_index + 3])
            qc.append(sub, [q1, q2])
            param_index += 3
        return qc

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            sub = conv_circuit(params[:3])
            qc.append(sub, [src, sink])
            params = params[3:]
        return qc

    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = [
    "FastBaseEstimator",
    "build_classifier_circuit",
    "Autoencoder",
    "QCNN",
]
