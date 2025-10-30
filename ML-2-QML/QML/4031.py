"""Quantum hybrid QCNN using EstimatorQNN with an additional regression circuit.

The quantum circuit constructs the convolution and pooling layers as in the
original QCNN.  After the ansatz, a small regression sub‑circuit (mirroring
EstimatorQNN) is appended on qubit 0.  The observable is a Pauli‑Y operator
on that qubit, yielding a scalar output suitable for regression tasks.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def HybridQCNN() -> EstimatorQNN:
    """Return a hybrid QCNN+regression QNN configured with EstimatorQNN."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # ------------ Convolution and Pooling primitives -----------------
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

    def pool_circuit(params):
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
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(param_vec[i * 3 : i * 3 + 3])
            qc.append(sub, [i, i + 1])
        return qc

    def pool_layer(sources, sinks, prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i, (src, snk) in enumerate(zip(sources, sinks)):
            sub = pool_circuit(param_vec[i * 3 : i * 3 + 3])
            qc.append(sub, [src, snk])
        return qc

    # ------------ Regression sub‑circuit --------------------------------
    def reg_circuit(params):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        return qc

    # ------------ Build full ansatz ------------------------------------
    ansatz = QuantumCircuit(8, name="QCNN_Ansatz")

    # First convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)

    # Second convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), list(range(4,8)), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)

    # Third convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), list(range(6,8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)

    # Regression circuit on qubit 0
    reg_params = ParameterVector("r", length=2)
    reg = reg_circuit(reg_params)
    ansatz.append(reg, [0])

    # Feature map
    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: Pauli‑Y on qubit 0
    observable = SparsePauliOp.from_list([("Y" + "I" * 7, 1)])

    # Assemble EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQCNN"]
