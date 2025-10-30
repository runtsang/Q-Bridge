import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNModel:
    """
    Quantum Convolutional Neural Network implemented as a Qiskit EstimatorQNN.
    The ansatz is a layered structure with explicit entanglement and pooling,
    mirroring the classical residual flow.
    """

    def __init__(self, num_qubits: int = 8) -> None:
        self.num_qubits = num_qubits
        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    # ---------- Circuit building helpers ----------
    def _conv_circuit(self, params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
        """Parameterized two‑qubit kernel."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    def _conv_layer(self, qubits: list[int], param_prefix: str) -> QuantumCircuit:
        """Single convolutional layer operating on pairs of qubits."""
        qc = QuantumCircuit(len(qubits), name="conv")
        params = ParameterVector(param_prefix, length=len(qubits) // 2 * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[idx : idx + 3], q1, q2)
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
        """Pooling kernel that discards one qubit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    def _pool_layer(
        self,
        sources: list[int],
        sinks: list[int],
        param_prefix: str,
    ) -> QuantumCircuit:
        """Pooling layer that measures and discards a qubit."""
        qc = QuantumCircuit(len(sources) + len(sinks), name="pool")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx : idx + 3], src, snk)
            qc.append(sub, [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full QCNN ansatz with 3 conv‑pool stages."""
        qc = QuantumCircuit(self.num_qubits)
        # Feature map
        qc.compose(self.feature_map, range(self.num_qubits), inplace=True)

        # Stage 1
        qc.compose(
            self._conv_layer(list(range(self.num_qubits)), "c1"),
            range(self.num_qubits),
            inplace=True,
        )
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 2)),
                list(range(self.num_qubits // 2, self.num_qubits)),
                "p1",
            ),
            range(self.num_qubits),
            inplace=True,
        )

        # Stage 2
        qc.compose(
            self._conv_layer(
                list(range(self.num_qubits // 2, self.num_qubits)), "c2"
            ),
            range(self.num_qubits // 2, self.num_qubits),
            inplace=True,
        )
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 4)),
                list(range(self.num_qubits // 4, self.num_qubits // 2)),
                "p2",
            ),
            range(self.num_qubits // 2, self.num_qubits),
            inplace=True,
        )

        # Stage 3
        qc.compose(
            self._conv_layer(
                list(range(self.num_qubits // 4, self.num_qubits // 2)), "c3"
            ),
            range(self.num_qubits // 4, self.num_qubits // 2),
            inplace=True,
        )
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 8)),
                list(range(self.num_qubits // 8, self.num_qubits // 4)),
                "p3",
            ),
            range(self.num_qubits // 4, self.num_qubits // 2),
            inplace=True,
        )

        return qc

    # ---------- Public API ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of feature vectors."""
        return self.qnn.predict(X)

    def get_parameters(self) -> np.ndarray:
        """Return current weight parameters."""
        return self.qnn.parameters

    def set_parameters(self, params: np.ndarray) -> None:
        """Set the weight parameters of the ansatz."""
        self.qnn.set_parameters(params)
