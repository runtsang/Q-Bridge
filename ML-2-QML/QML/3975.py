import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import algorithm_globals


def _build_qnn() -> EstimatorQNN:
    """Construct a QCNN with embedded quantum self‑attention sub‑circuits."""
    # Seed for reproducibility
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # --- Basic building blocks ------------------------------------------------
    def conv_unitary(params):
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

    def pool_unitary(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def attention_unitary(params, n_qubits):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            qc.cnot(i, i + 1)
        return qc

    # --- Layer constructors ---------------------------------------------------
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_unitary(params[idx : idx + 3])
            qc.append(sub.to_instruction(), [q1, q2])
            idx += 3
        return qc

    def pool_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = pool_unitary(params[idx : idx + 3])
            qc.append(sub.to_instruction(), [q1, q2])
            idx += 3
        return qc

    def attention_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        sub = attention_unitary(params, num_qubits)
        qc.append(sub.to_instruction(), range(num_qubits))
        return qc

    # --- Ansatz assembly -------------------------------------------------------
    ansatz = QuantumCircuit(8, name="HybridAnsatz")
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(attention_layer(8, "a1"), range(8), inplace=True)

    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(attention_layer(4, "a2"), range(4, 8), inplace=True)

    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), range(6, 8), inplace=True)
    ansatz.compose(attention_layer(2, "a3"), range(6, 8), inplace=True)

    # Feature map – classical encoding of the input
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


class QCNNSelfAttentionHybrid:
    """Quantum hybrid QCNN with embedded self‑attention sub‑circuits."""
    def __init__(self):
        self.qnn = _build_qnn()

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Run a forward pass using the underlying EstimatorQNN."""
        return self.qnn.predict(inputs)


__all__ = ["QCNNSelfAttentionHybrid"]
