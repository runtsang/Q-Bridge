import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def QCNNAttentionQML():
    estimator = Estimator()
    # Feature map
    fmap = ZFeatureMap(8)

    # Convolution and pooling subcircuits
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        return qc

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[i//2*3:(i//2+1)*3])
            qc.append(sub, [i, i+1])
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
        for i in range(0, num_qubits, 2):
            sub = pool_circuit(params[i//2*3:(i//2+1)*3])
            qc.append(sub, [i, i+1])
        return qc

    # Self‑attention subcircuit
    def self_attention_circuit(n_qubits, param_prefix):
        qc = QuantumCircuit(n_qubits)
        rot_params = ParameterVector(param_prefix+"_rot", length=n_qubits*3)
        ent_params = ParameterVector(param_prefix+"_ent", length=n_qubits-1)
        for i in range(n_qubits):
            qc.rx(rot_params[3*i], i)
            qc.ry(rot_params[3*i+1], i)
            qc.rz(rot_params[3*i+2], i)
        for i in range(n_qubits-1):
            qc.crx(ent_params[i], i, i+1)
        return qc

    # Build full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.append(conv_layer(8, "c1"), range(8))
    ansatz.append(pool_layer(8, "p1"), range(8))
    ansatz.append(conv_layer(4, "c2"), range(4,8))
    ansatz.append(pool_layer(4, "p2"), range(4,8))
    ansatz.append(conv_layer(2, "c3"), range(6,8))
    ansatz.append(pool_layer(2, "p3"), range(6,8))
    ansatz.append(self_attention_circuit(8, "attn"), range(8))

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.append(fmap, range(8))
    circuit.append(ansatz, range(8))

    # Observable
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=fmap.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNAttentionQML"]
