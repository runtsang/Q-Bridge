"""Hybrid quantum classifier module.

This module implements a quantum circuit that mirrors the classical network
defined in the ML counterpart.  It combines a data‑encoding Z‑feature map,
a QCNN ansatz built from convolutional and pooling blocks, and a small
quantum convolution filter (Ref. 4).  The circuit returns a binary
classification observable.

"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
import numpy as np
import qiskit

# --------------------------------------------------------------------------- #
# 1. Quantum building blocks
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        param_index += 3
    return qc


def _pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for src, sink in zip(sources, sinks):
        sub = _pool_circuit(params[param_index : param_index + 3])
        qc.append(sub, [src, sink])
        param_index += 3
    return qc


def _quantum_filter(kernel_size: int) -> QuantumCircuit:
    n = kernel_size ** 2
    qc = QuantumCircuit(n)
    theta = ParameterVector("theta", length=n)
    for i in range(n):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += QuantumCircuit.random(n, 2)
    return qc


# --------------------------------------------------------------------------- #
# 2. Circuit construction
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[BaseOperator]]:
    """Construct a hybrid quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of data qubits (must be a power of two for the QCNN ansatz).
    depth : int
        Number of layers in the QCNN ansatz.

    Returns
    -------
    circuit : QuantumCircuit
        The full variational circuit.
    encoding : list[Parameter]
        Parameters that encode the classical data.
    weights : list[Parameter]
        Variational parameters of the ansatz.
    observables : list[BaseOperator]
        Measurement operators that produce a binary output.
    """
    # 1. Feature map
    feature_map = ZFeatureMap(num_qubits)
    encoding = list(feature_map.parameters)

    # 2. Quantum filter (Ref. 4)
    filter_qc = _quantum_filter(kernel_size=2)

    # 3. QCNN ansatz
    qc = QuantumCircuit(num_qubits)
    qc.append(_conv_layer(num_qubits, "c1"), range(num_qubits))
    qc.append(_pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), range(num_qubits))
    qc.append(_conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2))
    qc.append(_pool_layer([0, 1], [2, 3], "p2"), range(num_qubits // 2))
    qc.append(_conv_layer(num_qubits // 4, "c3"), range(num_qubits // 4))
    qc.append(_pool_layer([0], [1], "p3"), range(num_qubits // 4))

    # 4. Combine feature map, filter, and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(filter_qc, list(range(filter_qc.num_qubits)), inplace=True)
    circuit.compose(qc, range(num_qubits), inplace=True)

    # 5. Observables
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    observables = [observable]

    # 6. Variational parameters
    weights = list(filter_qc.parameters) + [p for p in circuit.parameters if p.name.startswith("c") or p.name.startswith("p")]

    return circuit, encoding, weights, observables


# --------------------------------------------------------------------------- #
# 3. Estimator utilities
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: List[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: List[List[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class HybridQuantumClassifier:
    """Convenience wrapper that builds the circuit and exposes an evaluate method."""

    def __init__(self, num_qubits: int, depth: int):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def evaluate(self, parameter_sets: List[List[float]]) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(self.observables, parameter_sets)


__all__ = [
    "build_classifier_circuit",
    "FastBaseEstimator",
    "HybridQuantumClassifier",
]
