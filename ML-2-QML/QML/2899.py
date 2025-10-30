"""Quantum‑classical hybrid QCNN with a self‑attention block."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNNWithAttentionQNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # ---------- Helper circuits ----------
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

    def attention_circuit(params):
        """Simple self‑attention style block: single‑qubit rotations + entangling CRX."""
        qc = QuantumCircuit(2)
        # Three rotations per qubit
        for i in range(2):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        # Entangling layer
        qc.crx(params[6], 0, 1)
        return qc

    # ---------- Layer constructors ----------
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[i:i + 3]), [i, i+1])
        return qc

    def pool_layer(sources, sinks, prefix):
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, snk, p in zip(sources, sinks, params):
            qc.append(pool_circuit(p), [src, snk])
        return qc

    def attention_layer(qubits, prefix):
        qc = QuantumCircuit(len(qubits))
        params = ParameterVector(prefix, length=len(qubits) * 3 + 1)  # +1 for CRX angle
        for i, q in enumerate(qubits):
            qc.append(attention_circuit(params[3 * i:3 * i + 3]), [q])
        # Entangling across adjacent qubits
        for i in range(len(qubits) - 1):
            qc.crx(params[-1], qubits[i], qubits[i+1])
        return qc

    # ---------- Feature map ----------
    feature_map = ZFeatureMap(8)

    # ---------- Ansatz construction ----------
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolve & attention
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(attention_layer(list(range(8)), "a1"), inplace=True)

    # First pooling
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)

    # Second convolve & attention
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(attention_layer([0,1,2,3], "a2"), inplace=True)

    # Second pooling
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)

    # Third convolve & attention
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(attention_layer([0,1], "a3"), inplace=True)

    # Third pooling
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
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

__all__ = ["QCNNWithAttentionQNN"]
