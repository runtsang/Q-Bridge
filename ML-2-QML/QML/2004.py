"""Quantum Convolutional Neural Network (QCNN) implementation using Qiskit."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit import algorithm_globals


def QCNN() -> EstimatorQNN:
    """Return a 12‑qubit QCNN as an EstimatorQNN object."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        """Two‑qubit convolution kernel."""
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
        """Two‑qubit pooling kernel."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def conv_layer(num_qubits, param_prefix):
        """Convolutional layer acting on disjoint qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_circuit(params[idx: idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def pool_layer(sources, sinks, param_prefix):
        """Pooling layer reducing the number of active qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = pool_circuit(params[idx: idx + 3])
            qc.append(sub, [src, snk])
            idx += 3
        return qc

    # Build the ansatz
    ansatz = QuantumCircuit(12, name="QCNN Ansatz")

    # 1st Convolutional layer
    ansatz.compose(conv_layer(12, "c1"), range(12), inplace=True)

    # 1st Pooling layer (reduce to 6 qubits)
    sources1, sinks1 = list(range(0, 6)), list(range(6, 12))
    ansatz.compose(pool_layer(sources1, sinks1, "p1"), range(12), inplace=True)

    # 2nd Convolutional layer on the reduced 6‑qubit set
    ansatz.compose(conv_layer(6, "c2"), range(6), inplace=True)

    # 2nd Pooling layer (reduce to 3 qubits)
    sources2, sinks2 = list(range(0, 3)), list(range(3, 6))
    ansatz.compose(pool_layer(sources2, sinks2, "p2"), range(6), inplace=True)

    # 3rd Convolutional layer on the 3‑qubit set
    ansatz.compose(conv_layer(3, "c3"), range(3), inplace=True)

    # 3rd Pooling layer (reduce to 1 qubit)
    sources3, sinks3 = [0], [1]
    ansatz.compose(pool_layer(sources3, sinks3, "p3"), range(3), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(12)

    # Full circuit: feature map followed by ansatz
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)

    # Observable on the remaining qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 11, 1)])

    # Construct EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN"]
