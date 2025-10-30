import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNQuantum(EstimatorQNN):
    """
    Variational QCNN circuit built from convolution and pooling subcircuits.
    The circuit operates on 4 qubits and outputs a 4‑dimensional measurement
    vector. It is designed to accept a 4‑dimensional input vector from the
    classical branch of the hybrid model.
    """
    def __init__(self):
        estimator = StatevectorEstimator()

        # Convolution subcircuit operating on a pair of qubits
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

        # Convolution layer over all qubits
        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            param_vec = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(param_vec[i // 2 * 3 : i // 2 * 3 + 3])
                qc.compose(sub, [i, i + 1], inplace=True)
            return qc

        # Pooling subcircuit operating on a pair of qubits
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        # Pooling layer over all qubits
        def pool_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            param_vec = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                sub = pool_circuit(param_vec[i // 2 * 3 : i // 2 * 3 + 3])
                qc.compose(sub, [i, i + 1], inplace=True)
            return qc

        # Feature map that encodes the classical input into the quantum state
        feature_map = ZFeatureMap(4)

        # Ansatz consisting of one convolution‑pooling pair
        ansatz = QuantumCircuit(4)
        ansatz.compose(conv_layer(4, "c1"), inplace=True)
        ansatz.compose(pool_layer(4, "p1"), inplace=True)

        # Full circuit: feature map + ansatz
        circuit = QuantumCircuit(4)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observable: measure Z on all qubits
        observable = SparsePauliOp.from_list([("Z" * 4, 1)])

        super().__init__(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )
