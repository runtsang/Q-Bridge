"""Quantum QCNN with trainable feature map, parameter‑shift gradient and backend flexibility."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA


class QCNNHybrid:
    """Quantum convolutional neural network with a trainable ansatz."""
    def __init__(
        self,
        qubits: int = 8,
        conv_layers: int = 3,
        pool_layers: int = 3,
        backend: str = "statevector_simulator",
    ) -> None:
        self.qubits = qubits
        self.backend = backend
        self.estimator = Estimator(backend=backend)
        self._build_ansatz()
        self._build_feature_map()
        self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Feature map
    # ------------------------------------------------------------------
    def _build_feature_map(self) -> None:
        self.feature_map = ZFeatureMap(self.qubits, reps=1, entanglement="full")

    # ------------------------------------------------------------------
    # Convolution & pooling primitives
    # ------------------------------------------------------------------
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

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for i in range(0, num_qubits, 2):
            qc.append(self._conv_circuit(params[idx : idx + 3]), [i, i + 1])
            qc.barrier()
            idx += 3
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

    def _pool_layer(
        self,
        sources: list[int],
        sinks: list[int],
        prefix: str,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            qc.append(self._pool_circuit(params[idx : idx + 3]), [s, t])
            qc.barrier()
            idx += 3
        return qc

    # ------------------------------------------------------------------
    # Ansatz construction
    # ------------------------------------------------------------------
    def _build_ansatz(self) -> None:
        self.ansatz = QuantumCircuit(self.qubits, name="Ansatz")
        # First convolution
        self.ansatz.compose(self._conv_layer(self.qubits, "c1"), inplace=True)
        # First pooling
        self.ansatz.compose(
            self._pool_layer(
                list(range(self.qubits // 2)),
                list(range(self.qubits // 2, self.qubits)),
                "p1",
            ),
            inplace=True,
        )
        # Second convolution
        self.ansatz.compose(self._conv_layer(self.qubits // 2, "c2"), inplace=True)
        # Second pooling
        self.ansatz.compose(
            self._pool_layer(
                list(range(self.qubits // 4)),
                list(range(self.qubits // 4, self.qubits // 2)),
                "p2",
            ),
            inplace=True,
        )
        # Third convolution
        self.ansatz.compose(self._conv_layer(self.qubits // 4, "c3"), inplace=True)
        # Third pooling
        self.ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

    # ------------------------------------------------------------------
    # Full circuit assembly
    # ------------------------------------------------------------------
    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient via the parameter‑shift rule."""
        grad = np.zeros(len(self.ansatz.parameters))
        shift = np.pi / 2
        for i, param in enumerate(self.ansatz.parameters):
            plus = np.array(self.ansatz.parameters, copy=True)
            minus = np.array(self.ansatz.parameters, copy=True)
            plus[i] += shift
            minus[i] -= shift
            self.qnn.set_weights(plus)
            f_plus = self.qnn.predict(X)
            self.qnn.set_weights(minus)
            f_minus = self.qnn.predict(X)
            grad[i] = np.mean((f_plus - f_minus) * 2 * (f_plus - y) / shift)
        return grad

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        opt_type: str = "cobyla",
    ) -> None:
        """Hybrid training loop."""
        if opt_type.lower() == "spsa":
            opt = SPSA(maxiter=epochs)
        else:
            opt = COBYLA(maxiter=epochs)

        def loss(weights):
            self.qnn.set_weights(weights)
            preds = self.qnn.predict(X)
            return np.mean((preds - y) ** 2)

        init = np.random.rand(len(self.ansatz.parameters))
        opt.optimize(init, loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.qnn.predict(X)

    def export_qiskit_circuit(self) -> QuantumCircuit:
        """Return the full circuit for further Qiskit manipulation."""
        return self.circuit
