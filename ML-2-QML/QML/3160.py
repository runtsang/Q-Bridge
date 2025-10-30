import json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNQuanvolutionHybridQNN:
    """
    Quantum‑only counterpart to the hybrid class.  It implements a QCNN
    convolution‑pooling stack that directly consumes a 2‑D image encoded
    into qubits via a Z‑feature map.  The architecture mirrors the
    classical pipeline: a feature‑map → convolutional layers → pooling
    layers → final measurement.  The circuit is built with reusable
    sub‑circuits for convolution and pooling, and the parameters are
    trainable through an EstimatorQNN wrapper.
    """

    def __init__(self, num_qubits: int = 8, num_params: int = 48):
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.num_qubits = num_qubits
        self.num_params = num_params
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        # Feature map: 8‑qubit ZFeatureMap
        feature_map = self._feature_map()

        # Convolutional and pooling layers
        ansatz = QuantumCircuit(self.num_qubits)
        # First conv layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        # First pooling
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(self.num_qubits), inplace=True)
        # Second conv layer
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), range(self.num_qubits // 2, self.num_qubits), inplace=True)
        # Second pooling
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(self.num_qubits // 2, self.num_qubits), inplace=True)
        # Third conv layer
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), range(self.num_qubits // 4, self.num_qubits // 2), inplace=True)
        # Third pooling
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(self.num_qubits // 4, self.num_qubits // 2), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        return circuit.decompose()

    def _feature_map(self) -> QuantumCircuit:
        # 8‑qubit ZFeatureMap
        fm = QuantumCircuit(self.num_qubits)
        for q in range(self.num_qubits):
            fm.rz(np.pi * 0.5, q)  # simple rotation for illustration
        return fm

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i*3:(i+1)*3])
            qc.append(sub, [i, i+1])
            qc.barrier()
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        c = QuantumCircuit(2)
        c.rz(-np.pi/2, 1)
        c.cx(1, 0)
        c.rz(params[0], 0)
        c.ry(params[1], 1)
        c.cx(0, 1)
        c.ry(params[2], 1)
        c.cx(1, 0)
        c.rz(np.pi/2, 0)
        return c

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num_pairs = len(sources)
        qc = QuantumCircuit(num_pairs * 2)
        params = ParameterVector(prefix, length=num_pairs * 3)
        for idx, (src, snk) in enumerate(zip(sources, sinks)):
            sub = self._pool_circuit(params[idx*3:(idx+1)*3])
            qc.append(sub, [src, snk])
            qc.barrier()
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        p = QuantumCircuit(2)
        p.rz(-np.pi/2, 1)
        p.cx(1, 0)
        p.rz(params[0], 0)
        p.ry(params[1], 1)
        p.cx(0, 1)
        p.ry(params[2], 1)
        return p

    def get_qnn(self) -> EstimatorQNN:
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        return EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self._feature_map().parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator
        )
__all__ = ["QCNNQuanvolutionHybridQNN"]
