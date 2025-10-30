import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQuantumLayer:
    """Hybrid quantum layer built from QCNN and QuantumNAT inspired ansatz."""
    def __init__(self):
        self.qnn = self._build_qnn()

    def _build_qnn(self):
        # Feature map
        feature_map = QuantumCircuit(8)
        feature_map.h(range(8))

        # Convolutional sub‑circuit
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1,0)
            qc.rz(params[0],0)
            qc.ry(params[1],1)
            qc.cx(0,1)
            qc.ry(params[2],1)
            qc.cx(1,0)
            qc.rz(np.pi/2,0)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits//2*3)
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(params[i*3:(i+3)])
                qc.append(sub, [i,i+1])
                qc.barrier()
            return qc

        # Pooling sub‑circuit
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2,1)
            qc.cx(1,0)
            qc.rz(params[0],0)
            qc.ry(params[1],1)
            qc.cx(0,1)
            qc.ry(params[2],1)
            return qc

        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits//2*3)
            for i in range(0, num_qubits, 2):
                sub = pool_circuit(params[i*3:(i+3)])
                qc.append(sub, [i,i+1])
                qc.barrier()
            return qc

        # Quantum‑NAT inspired random‑rotation layer
        def quantum_nature_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits)
            for i in range(num_qubits):
                qc.ry(params[i], i)
                qc.cx(i, (i+1)%num_qubits)
            return qc

        # Assemble ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8,"c1"), inplace=True)
        ansatz.compose(pool_layer(8,"p1"), inplace=True)
        ansatz.compose(conv_layer(4,"c2"), inplace=True)
        ansatz.compose(pool_layer(4,"p2"), inplace=True)
        ansatz.compose(conv_layer(2,"c3"), inplace=True)
        ansatz.compose(pool_layer(2,"p3"), inplace=True)
        ansatz.compose(quantum_nature_layer(8,"qn"), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z"+"I"*7, 1)])

        estimator = Estimator()
        return EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )

    def evaluate(self, input_values, weight_values, shots=1024, seed=None):
        return self.qnn.evaluate(input_values, weight_values, shots=shots, seed=seed)

def FCL():
    """Factory returning the hybrid quantum layer."""
    return HybridQuantumLayer()

__all__ = ["HybridQuantumLayer", "FCL"]
