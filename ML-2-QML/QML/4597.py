"""
Hybrid quantum‑classical classifier – quantum side.

This module implements a QCNN‑style ansatz followed by a variational
readout.  The ``build_classifier_circuit`` factory returns
a Qiskit circuit, parameter metadata and the observable for
probability estimation, ready to be wrapped in an EstimatorQNN.
"""

from __future__ import annotations

import numpy as np
import math
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Helper functions – QCNN convolution / pooling layers
# --------------------------------------------------------------------------- #

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single 2‑qubit convolution unit used by QCNN layers."""
    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi / 2, 1)
    circuit.cx(1, 0)
    circuit.rz(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 1)
    circuit.cx(1, 0)
    circuit.rz(np.pi / 2, 0)
    return circuit

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single 2‑qubit pooling unit used by QCNN layers."""
    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi / 2, 1)
    circuit.cx(1, 0)
    circuit.rz(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 1)
    return circuit

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer over a chain of qubits."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        param_index += 3
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that reduces the qubit count."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    param_index = 0
    params = ParameterVector(param_prefix, length=(len(sources) // 2) * 3)
    for src, snk in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]), [src, snk], inplace=True)
        param_index += 3
    return qc

# --------------------------------------------------------------------------- #
# Main factory
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int = 8,
    depth: int = 2,
    estimator: StatevectorEstimator | None = None,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Constructs a QCNN‑style variational circuit and returns
    the circuit, encoding params, weight params and the read‑out observable.

    Parameters
    ----------
    num_qubits : int
        Total qubits used in the QCNN ansatz (default 8).
    depth : int
        Number of convolution/pooling cycles.
    estimator : StatevectorEstimator, optional
        If supplied, the circuit will be wrapped in an EstimatorQNN for
        immediate evaluation.

    Returns
    -------
    circuit : QuantumCircuit
        The full variational circuit (feature map + ansatz).
    encoding : Iterable[ParameterVector]
        Parameters used for the data‑encoding feature map.
    weight_params : Iterable[ParameterVector]
        Parameters of the variational ansatz.
    observables : List[SparsePauliOp]
        Read‑out observable(s).
    """
    # Feature map – 8‑qubit ZFeatureMap
    feature_map = ZFeatureMap(num_qubits)
    encoding = feature_map.parameters

    # Construct the QCNN ansatz
    ansatz = QuantumCircuit(num_qubits, name="QCNN_Ansatz")
    # First convolution and pooling
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"), range(num_qubits), inplace=True)
    # Second convolution and pooling
    ansatz.compose(conv_layer(num_qubits//2, "c2"), range(num_qubits//2, num_qubits), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), range(num_qubits//2, num_qubits), inplace=True)
    # Third convolution and pooling
    ansatz.compose(conv_layer(num_qubits//4, "c3"), range(num_qubits//4*3, num_qubits), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(num_qubits//4*3, num_qubits), inplace=True)

    weight_params = ansatz.parameters

    # Observable – measure Z on the first qubit only
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    # Full circuit: feature map + ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    if estimator is None:
        estimator = StatevectorEstimator()

    # Optional: wrap in EstimatorQNN for direct gradient estimation
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=encoding,
        weight_params=weight_params,
        estimator=estimator,
    )

    return circuit, encoding, weight_params, [observable]

__all__ = ["build_classifier_circuit"]
