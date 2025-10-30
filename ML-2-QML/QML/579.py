import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from torch.optim import Adam


class EstimatorQNN:
    """A hybrid variational circuit with entanglement and parameter‑shift training.
    The circuit is built for `num_qubits` qubits and `depth` layers of entangling gates.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        estimator: StatevectorEstimator | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.estimator = estimator or StatevectorEstimator()
        self.circuit = self._build_circuit()
        self.input_params = [p for p in self.circuit.parameters if "x_" in p.name]
        self.weight_params = [p for p in self.circuit.parameters if "w_" in p.name]
        self.model = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self._build_observable(),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a depth‑controlled entangling variational circuit."""
        qc = QuantumCircuit(self.num_qubits)
        # Parameterised rotations for each qubit
        for q in range(self.num_qubits):
            alpha = Parameter(f"x_{q}")  # input parameter
            beta = Parameter(f"w_{q}")   # weight parameter
            qc.ry(alpha, q)
            qc.rx(beta, q)
        # Entangling layers
        for d in range(self.depth):
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(self.num_qubits):
                gamma = Parameter(f"w_{q}_{d}")  # depth‑dependent weight
                qc.rz(gamma, q)
        return qc

    def _build_observable(self) -> SparsePauliOp:
        """Observable as tensor product of Z on all qubits."""
        return SparsePauliOp.from_list([("Z" * self.num_qubits, 1)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: map classical inputs to quantum expectation values."""
        param_dict = {}
        for i, p in enumerate(self.input_params):
            param_dict[p] = X[:, i % X.shape[1]]
        return self.model.predict(param_dict).reshape(-1, 1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Train the quantum circuit using the built‑in optimizer."""
        if self.model.optimizer is None:
            self.model.optimizer = Adam
        self.model.fit(X, y, epochs=epochs, lr=lr)


__all__ = ["EstimatorQNN"]
