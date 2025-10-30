import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import ZFeatureMap

def HybridEstimatorQNN():
    """Construct a variational quantum neural network with a convolution‑style ansatz."""
    # Feature map
    num_qubits = 4
    feature_map = ZFeatureMap(num_qubits, reps=1, insert_barriers=False)

    # Convolution and pooling sub‑circuits
    def conv_circuit(params):
        qc = QuantumCircuit(num_qubits)
        qc.rz(-np.pi/2, 0)
        qc.cx(0,1)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(1,0)
        qc.ry(params[2], 0)
        qc.cx(0,1)
        qc.rz(np.pi/2, 1)
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(num_qubits)
        qc.rz(-np.pi/2, 1)
        qc.cx(1,0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0,1)
        qc.ry(params[2], 0)
        return qc

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits*3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(param_vec[i*3:(i+2)*3])
            qc.append(sub, [i, i+1])
        return qc

    def pool_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits*3)
        for i in range(0, num_qubits, 2):
            sub = pool_circuit(param_vec[i*3:(i+2)*3])
            qc.append(sub, [i, i+1])
        return qc

    # Build full ansatz
    ansatz = QuantumCircuit(num_qubits)
    ansatz.append(conv_layer(num_qubits, "c1"), range(num_qubits))
    ansatz.append(pool_layer(num_qubits, "p1"), range(num_qubits))
    ansatz.append(conv_layer(num_qubits//2, "c2"), range(num_qubits//2))
    ansatz.append(pool_layer(num_qubits//2, "p2"), range(num_qubits//2))
    ansatz.append(conv_layer(num_qubits//4, "c3"), range(num_qubits//4))
    ansatz.append(pool_layer(num_qubits//4, "p3"), range(num_qubits//4))

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.append(feature_map, range(num_qubits))
    circuit.append(ansatz, range(num_qubits))

    observable = SparsePauliOp.from_list([("Z"*num_qubits, 1)])
    estimator = StatevectorEstimator()

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridEstimatorQNN"]
