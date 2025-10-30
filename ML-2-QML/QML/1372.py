"""Hybrid QCNN quantum neural network with configurable feature map and ansatz.

The class encapsulates the construction of a QCNN circuit and exposes a
`get_qnn` method that returns an EstimatorQNN instance ready for training.
It supports:
- Choice of feature map (ZFeatureMap, PauliFeatureMap).
- Automatic expansion of the ansatz into convolutional and pooling layers.
- Optional use of Pennylane's `qml.QNode` for differentiable execution.
"""

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNHybrid:
    """Quantum QCNN with configurable feature map and ansatz."""

    def __init__(
        self,
        num_qubits: int = 8,
        feature_map_cls: type = ZFeatureMap,
        feature_map_params: dict | None = None,
        seed: int | None = 12345,
    ):
        self.num_qubits = num_qubits
        self.feature_map_cls = feature_map_cls
        self.feature_map_params = feature_map_params or {}
        self.seed = seed
        algorithm_globals.random_seed = seed  # set global seed for reproducibility
        self.estimator = Estimator()
        self.circuit = self._build_circuit()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _layer(self, layer_type: str, qubits: list[int], prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(qubits))
        param_len = len(qubits) // 2 * 3
        params = ParameterVector(prefix, length=param_len)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            circ = (
                self._conv_circuit(params[idx : idx + 3])
                if layer_type == "conv"
                else self._pool_circuit(params[idx : idx + 3])
            )
            qc.compose(circ, [q1, q2], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        feature_map = self.feature_map_cls(self.num_qubits, **self.feature_map_params)
        ansatz = QuantumCircuit(self.num_qubits)

        # First convolutional layer
        ansatz.compose(self._layer("conv", list(range(self.num_qubits)), "c1"), inplace=True)
        # First pooling layer
        ansatz.compose(self._layer("pool", list(range(self.num_qubits)), "p1"), inplace=True)
        # Second convolutional layer (on the reduced qubit set)
        ansatz.compose(
            self._layer("conv", list(range(self.num_qubits // 2, self.num_qubits)), "c2"),
            inplace=True,
        )
        # Second pooling layer
        ansatz.compose(
            self._layer("pool", list(range(self.num_qubits // 2, self.num_qubits)), "p2"),
            inplace=True,
        )
        # Third convolutional layer
        ansatz.compose(
            self._layer("conv", list(range(self.num_qubits // 4, self.num_qubits)), "c3"),
            inplace=True,
        )
        # Third pooling layer
        ansatz.compose(
            self._layer("pool", list(range(self.num_qubits // 4, self.num_qubits)), "p3"),
            inplace=True,
        )

        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit.decompose()

    def get_qnn(self) -> EstimatorQNN:
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.feature_map_cls(self.num_qubits).parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        return qnn

    def get_pennylane_qnode(self, device_name: str = "default.qubit") -> qml.QNode:
        dev = qml.device(device_name, wires=self.num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit_fn(inputs, weights):
            # Feature map: simple RX rotations
            for i, w in enumerate(inputs):
                qml.RX(w, wires=i)
            # Ansatz: a few parametric layers
            idx = 0
            for layer in range(3):
                for q in range(self.num_qubits // (2**layer)):
                    qml.RZ(weights[idx], wires=q)
                    idx += 1
            return qml.expval(qml.PauliZ(0))

        return circuit_fn

__all__ = ["QCNNHybrid"]
