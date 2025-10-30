import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


def QCNNNet() -> EstimatorQNN:
    """Quantum QCNN with a deeper, entangled ansatz.

    The circuit consists of a ZZFeatureMap to embed the data, followed by
    three convolution‑pooling stages built from RealAmplitudes blocks.
    Each convolution applies pairwise entanglement across the qubits,
    and pooling discards half of the qubits via a parametric two‑qubit
    circuit. The final measurement is a single Z observable on the
    first qubit, providing a scalar output suitable for binary
    classification. The ansatz parameters are optimised with COBYLA.
    """
    # Feature map
    feature_map = ZZFeatureMap(num_qubits=8, reps=2, entanglement='full')

    # Convolution block: RealAmplitudes with full entanglement
    def conv_block(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.compose(RealAmplitudes(2, entanglement='full', reps=1), [0, 1], inplace=True)
        return qc

    # Pooling block: two‑qubit unitary that discards one qubit
    def pool_block(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.compose(RealAmplitudes(2, entanglement='full', reps=1), [0, 1], inplace=True)
        return qc

    # Assemble convolutional layer
    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.compose(conv_block(params[idx : idx + 3]), [q1, q2], inplace=True)
            idx += 3
        return qc

    # Assemble pooling layer
    def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.compose(pool_block(params[idx : idx + 3]), [q1, q2], inplace=True)
            idx += 3
        return qc

    # Build the full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: single Z on qubit 0
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator
    estimator = Estimator()

    # Construct the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    return qnn


__all__ = ["QCNNNet"]
