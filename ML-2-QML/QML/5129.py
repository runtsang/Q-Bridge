import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

class HybridQCNN:
    """Quantum implementation of the HybridQCNN architecture.

    The circuit mirrors the classical hybrid model: a feature map encodes the input,
    followed by convolutional and pooling layers implemented with parameterized
    two‑qubit blocks.  The resulting circuit is wrapped in an EstimatorQNN so it
    can be trained with classical optimizers.  The class also exposes a ``kernel_matrix``
    method that evaluates the quantum kernel between two data sets using the
    same ansatz.
    """
    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)
        self.circuit = self._build_ansatz()
        self.observables = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _conv_block(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_block(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for i in range(0, num_qubits, 2):
            block = self._conv_block(params[idx: idx + 3])
            qc.append(block, [i, i + 1])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=len(sources) // 2 * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            block = self._pool_block(params[idx: idx + 3])
            qc.append(block, [src, sink])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # First convolution + pooling
        qc.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        qc.compose(
            self._pool_layer(
                list(range(self.n_qubits // 2)),
                list(range(self.n_qubits // 2, self.n_qubits)),
                "p1",
            ),
            inplace=True,
        )
        # Second convolution + pooling
        qc.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        qc.compose(
            self._pool_layer(
                list(range(self.n_qubits // 4)),
                list(range(self.n_qubits // 4, self.n_qubits // 2)),
                "p2",
            ),
            inplace=True,
        )
        # Third convolution + pooling
        qc.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return the QNN prediction for the given classical input."""
        return self.qnn.predict(inputs)

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two data sets using the quantum kernel."""
        # Use the same ansatz and observable as the QNN
        qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        return qnn.kernel_matrix(a, b)

__all__ = ["HybridQCNN"]
