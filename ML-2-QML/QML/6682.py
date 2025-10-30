"""Quantum QCNN with incremental data‑uploading and convolution‑pooling.

This circuit combines the convolutional layers from the QCNN
example with the incremental encoding strategy from the
QuantumClassifierModel.  The result is a variational
ansatz that can be trained end‑to‑end with a state‑vector
estimator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from typing import List

def QCNNHybrid(num_qubits: int = 8, depth: int = 3) -> EstimatorQNN:
    """
    Build a QCNN circuit with depth‑controlled convolution‑pooling
    layers and an incremental Rx encoding of the input data.
    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the feature map size).
    depth : int
        Number of convolution‑pooling pairs.
    Returns
    -------
    EstimatorQNN
        Variational quantum neural network ready for training.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

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

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(conv_circuit(params[idx:idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def pool_layer(sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.append(pool_circuit(params[idx:idx + 3]), [src, snk])
            qc.barrier()
            idx += 3
        return qc

    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Encoding circuit (incremental Rx)
    encoding = ParameterVector("x", num_qubits)
    encoding_circuit = QuantumCircuit(num_qubits)
    for q, p in enumerate(encoding):
        encoding_circuit.rx(p, q)

    # Ansatz construction
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(encoding_circuit, range(num_qubits), inplace=True)

    for d in range(depth):
        ansatz.compose(conv_layer(num_qubits, f"c{d+1}"), range(num_qubits), inplace=True)
        srcs = list(range(0, num_qubits, 2))
        snks = list(range(1, num_qubits, 2))
        ansatz.compose(pool_layer(srcs, snks, f"p{d+1}"), range(num_qubits), inplace=True)

    # Combine feature map and ansatz
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, range(num_qubits), inplace=True)
    full_circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observables: one Z per qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    # Separate encoding parameters from variational weights
    encoding_params = list(encoding.parameters)
    weight_params = [p for p in ansatz.parameters if p not in encoding_params]

    qnn = EstimatorQNN(
        circuit=full_circuit.decompose(),
        observables=observables,
        input_params=list(feature_map.parameters) + encoding_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNHybrid"]
