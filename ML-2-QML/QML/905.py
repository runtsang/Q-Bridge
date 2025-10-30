import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_aer.noise import NoiseModel

class QCNNModel:
    """
    Quantum Convolutional Neural Network.
    Builds a modular ansatz with convolution and pooling layers.
    Supports noise‑aware execution and hybrid training via parameter‑shift.
    """
    def __init__(self, num_qubits: int = 8,
                 conv_depth: int = 3,
                 pool_depth: int = 3,
                 backend=None,
                 noise_model: NoiseModel | None = None,
                 seed: int | None = 12345):
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.noise_model = noise_model
        self.seed = seed
        self.circuit = self._build_circuit()
        self.estimator = Estimator(backend=self.backend,
                                   noise_model=self.noise_model,
                                   seed_simulator=self.seed)
        self.qnn = EstimatorQNN(circuit=self.circuit.decompose(),
                                observables=self._observable(),
                                input_params=self.feature_map.parameters,
                                weight_params=self.ansatz.parameters,
                                estimator=self.estimator)

    def _observable(self) -> SparsePauliOp:
        return SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

    def _feature_map(self) -> QuantumCircuit:
        self.feature_map = ZFeatureMap(self.num_qubits)
        return self.feature_map

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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx:idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int],
                    param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx + 3])
            qc.append(sub, [src, snk])
            idx += 3
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        self.feature_map = self._feature_map()
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)

        # First Pooling Layer
        ansatz.compose(self._pool_layer(list(range(self.num_qubits // 2)),
                                        list(range(self.num_qubits // 2, self.num_qubits)),
                                        "p1"), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)

        # Second Pooling Layer
        ansatz.compose(self._pool_layer(list(range(self.num_qubits // 4)),
                                        list(range(self.num_qubits // 4, self.num_qubits // 2)),
                                        "p2"), inplace=True)

        # Third Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)

        # Third Pooling Layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        self.ansatz = ansatz
        return circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass returning expectation values for each input sample.
        """
        param_values = {p: val for p, val in zip(self.feature_map.parameters, X.T)}
        results = self.estimator.run(
            circuits=[self.circuit],
            parameter_values=[param_values]
        )
        return results[0].values.real

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 10,
              learning_rate: float = 0.01,
              optimizer_cls=COBYLA,
              seed: int | None = None):
        """
        Simple hybrid training loop using the chosen classical optimizer.
        """
        opt = optimizer_cls(maxiter=epochs, tol=1e-6, disp=False)
        weight_shape = self.ansatz.num_parameters
        weights = np.random.randn(weight_shape) * 0.01
        for epoch in range(epochs):
            def loss_fn(params):
                self.estimator.set_parameter_values(list(zip(self.ansatz.parameters, params)))
                preds = self.predict(X)
                loss = np.mean((preds - y) ** 2)
                return loss
            weights = opt.minimize(loss_fn, weights).x
        self.estimator.set_parameter_values(list(zip(self.ansatz.parameters, weights)))
        return weights

def QCNN(**kwargs):
    return QCNNModel(**kwargs)

__all__ = ["QCNNModel", "QCNN"]
