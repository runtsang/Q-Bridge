import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNN:
    """Quantum QCNN with convolution, pooling, and random layers."""
    def __init__(self):
        self.qnn = self._build_qnn()

    # ----- Sub‑circuits ----------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _random_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            qc.ry(params[3*i], i)
            qc.rz(params[3*i+1], i)
            qc.rx(params[3*i+2], i)
        return qc

    # ----- Layer combiners ------------------------------------------------
    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i*3:(i+2)*3])
            qc.append(sub, [i, i+1])
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=len(sources) * 3)
        for s, t, p in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[p*3:(p+1)*3])
            qc.append(sub, [s, t])
        return qc

    # ----- QCNN construction -----------------------------------------------
    def _build_qnn(self) -> EstimatorQNN:
        estimator = Estimator()

        # Simple Z‑feature map
        feature_map = QuantumCircuit(8)
        for i in range(8):
            feature_map.ry(ParameterVector("x"+str(i), 1)[0], i)

        # Ansatz with conv, random, and pool layers
        ansatz = QuantumCircuit(8)
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._random_layer(8, "r1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)

        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._random_layer(4, "r2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)

        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._random_layer(2, "r3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

    # ----- Forward ---------------------------------------------------------
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on the given input data.
        Args:
            inputs: Numpy array of shape (batch, 8) with feature values.
        Returns:
            Numpy array of expectation values.
        """
        return self.qnn.predict(inputs)
