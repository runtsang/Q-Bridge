import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RandomLayer
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class HybridQCNN:
    """
    Quantum hybrid QCNN that mirrors the classical architecture.

    * Feature map (ZFeatureMap) encodes the 8‑dimensional classical
      input into a 4‑qubit state.
    * Convolution & pooling layers are built from the QCNN template.
    * A RandomLayer and a set of parameterised gates (RX, RY, RZ, CRX)
      emulate the QLayer from Quantum‑NAT.
    * The circuit returns a 4‑dimensional expectation value vector.
    """
    def __init__(self):
        algorithm_globals.random_seed = 12345

    def _conv_circuit(self, params):
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

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(params[param_index:param_index+3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    def build(self):
        estimator = Estimator()

        # Feature map from classical input to 4 qubits
        feature_map = ZFeatureMap(4)

        # Construction of the ansatz
        ansatz = QuantumCircuit(4, name="Ansatz")

        # Convolution & pooling stages (QCNN style)
        ansatz.compose(self._conv_layer(4, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p1"), inplace=True)

        # Random layer (QuantumNAT style)
        random_layer = RandomLayer(n_ops=20, wires=[0,1,2,3])
        ansatz.compose(random_layer, inplace=True)

        # Parameterised gates (QuantumNAT style)
        rx = ParameterVector("rx", length=4)
        ry = ParameterVector("ry", length=4)
        rz = ParameterVector("rz", length=4)
        crx = ParameterVector("crx", length=4)
        for i in range(4):
            ansatz.rx(rx[i], i)
            ansatz.ry(ry[i], i)
            ansatz.rz(rz[i], i)
        for i in range(4):
            ansatz.crx(crx[i], [i, (i+1) % 4])

        ansatz.h(3)
        ansatz.sx(2)
        ansatz.cx(3, 0)

        # Assemble full circuit
        circuit = QuantumCircuit(4)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])

        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

def HybridQCNNFactory():
    """Return an instance of the quantum hybrid model."""
    return HybridQCNN().build()

__all__ = ["HybridQCNNFactory"]
