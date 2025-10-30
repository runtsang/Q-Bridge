import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def HybridQCNN() -> EstimatorQNN:
    """Constructs a hybrid QCNN quantum neural network.

    The circuit implements three convolutional layers followed by pooling,
    mirroring the classical architecture in :func:`HybridQCNN` but realized
    with parameter‑shared two‑qubit blocks.  A :class:`~qiskit.primitives.StatevectorEstimator`
    evaluates the circuit and the resulting :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`
    can be trained with classical optimizers.
    """

    # --- Convolution block -------------------------------------------------
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

    # --- Pooling block -----------------------------------------------------
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # --- Layer composition helpers ----------------------------------------
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for q in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[idx:idx+3]), [q, q+1])
            idx += 3
        return qc

    def pool_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q in range(0, num_qubits, 2):
            qc.append(pool_circuit(params[idx:idx+3]), [q, q+1])
            idx += 3
        return qc

    # --- Feature map -------------------------------------------------------
    feature_map = ZFeatureMap(8)

    # --- Ansatz construction -----------------------------------------------
    ansatz = QuantumCircuit(8, name="QCNN Ansatz")

    # First conv + pool
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)

    # Second conv + pool (reduced qubit count after pooling)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)

    # Third conv + pool (final 2 qubits)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for measurement
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap with EstimatorQNN
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQCNN", "EstimatorQNN"]
