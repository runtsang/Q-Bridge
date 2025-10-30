"""Hybrid fast estimator for Qiskit circuits.

The class ``HybridFastEstimator`` mirrors the classical version but operates on
parametrized QuantumCircuit objects.  It supports exact Statevector evaluation
and optional shot‑noise simulation.  The implementation is deliberately
light‑weight so that it can be used as a drop‑in replacement for the legacy
FastBaseEstimator in quantum workflows.

Features
--------
* Supports arbitrary parametrized circuits.
* Can evaluate a list of observables (BaseOperator or Pauli strings).
* Optional shot‑noise via a simple Gaussian model.
* Compatibility with EstimatorQNN for variational circuits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> Sequence[float]:
    """Return a sequence of values; if a single float is passed, wrap it."""
    if isinstance(values, (float, int)):
        return [float(values)]
    return list(values)

# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridFastEstimator:
    """Evaluate a parametrized QuantumCircuit or EstimatorQNN."""

    def __init__(self, circuit: QuantumCircuit | EstimatorQNN, *, shots: int | None = None, seed: int | None = None) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit | EstimatorQNN
            The circuit to evaluate.  If an EstimatorQNN is passed, the
            underlying Estimator will be used for expectation values.
        shots : int | None, optional
            If provided, Gaussian shot noise with variance ``1/shots`` is added.
        seed : int | None, optional
            Random seed for reproducibility of the noise.
        """
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._estimator = None
        if isinstance(circuit, EstimatorQNN):
            self._estimator = circuit.estimator

    # ------------------------------------------------------------------ #
    def _evaluate_statevector(
        self,
        observables: Iterable[BaseOperator],
        params: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Exact Statevector evaluation for a list of parameter sets."""
        results: List[List[complex]] = []
        for p in params:
            bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, p)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------ #
    def _evaluate_estimatorqnn(
        self,
        observables: Iterable[BaseOperator],
        params: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluation using EstimatorQNN."""
        # EstimatorQNN expects input_params and weight_params separately.
        # For simplicity, we assume all parameters are weight parameters.
        # The user must provide the observables and input data separately
        # if a feature map is required.
        results: List[List[complex]] = []
        for p in params:
            # The EstimatorQNN predict method returns a numpy array of shape (1,).
            # We need to map the result to the observables.
            # For now, we assume a single observable.
            pred = self.circuit.predict(p)
            results.append([complex(pred[0])])
        return results

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate observables for each parameter set."""
        if isinstance(self.circuit, EstimatorQNN):
            raw = self._evaluate_estimatorqnn(observables, parameter_sets)
        else:
            raw = self._evaluate_statevector(observables, parameter_sets)

        if self.shots is None:
            return raw

        rng = np.random.default_rng(self.seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(val.real, max(1e-6, 1 / self.shots))
                ) + 1j * rng.normal(val.imag, max(1e-6, 1 / self.shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit, shots: int | None = None, seed: int | None = None) -> "HybridFastEstimator":
        """Convenience constructor for a plain QuantumCircuit."""
        return cls(circuit, shots=shots, seed=seed)

    @classmethod
    def from_estimatorqnn(cls, qnn: EstimatorQNN, shots: int | None = None, seed: int | None = None) -> "HybridFastEstimator":
        """Convenience constructor for an EstimatorQNN."""
        return cls(qnn, shots=shots, seed=seed)

# --------------------------------------------------------------------------- #
# Example circuit factories (QCNN / Quantum‑NAT)
# --------------------------------------------------------------------------- #
def QCNN(num_qubits: int = 8) -> QuantumCircuit:
    """Return a QCNN circuit built with convolution and pooling layers."""
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZFeatureMap

    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            param_index += 3
        return qc

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.append(pool_circuit(params[param_index : param_index + 3]), [src, snk])
            param_index += 3
        return qc

    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit


def QuantumNAT(num_wires: int = 4) -> QuantumCircuit:
    """Return a simple Quantum‑NAT inspired circuit with random layers."""
    from qiskit.circuit import ParameterVector
    qc = QuantumCircuit(num_wires)
    params = ParameterVector("θ", length=num_wires * 4)
    for i in range(num_wires):
        qc.rx(params[i * 4], i)
        qc.ry(params[i * 4 + 1], i)
        qc.rz(params[i * 4 + 2], i)
        qc.cx(i, (i + 1) % num_wires)
        qc.rz(params[i * 4 + 3], i)
    return qc


__all__ = ["HybridFastEstimator", "QCNN", "QuantumNAT"]
