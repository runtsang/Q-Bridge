"""Quantum estimator with a 3‑qubit variational circuit.

The seed used a single‑qubit circuit.  This version expands the ansatz to
three qubits, adds entangling gates, and measures a multi‑qubit observable.
It still uses qiskit_machine_learning's EstimatorQNN class, but the
circuit and observables are richer, allowing more expressive quantum
regression.

The class name EstimatorQNN matches the classical counterpart so that
experiments can be run side‑by‑side.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN:
    """Quantum neural network estimator with 3‑qubit ansatz.

    Architecture:
        - 3 qubits
        - Parameterised Ry rotations on each qubit for input encoding
        - Parameterised Rz rotations on each qubit as trainable weights
        - Entanglement ladder (CX 0→1, CX 1→2)
        - Measurement of Y⊗Z⊗X observable
    """

    def __init__(self, input_dim: int = 2):
        # Define parameters
        self.input_params = [Parameter(f"inp_{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(input_dim)]

        # Build circuit
        qc = QuantumCircuit(3)
        # Input encoding
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # Parameterised rotation layer
        for i, p in enumerate(self.weight_params):
            qc.rz(p, i)
        # Entanglement
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Observable: Y⊗Z⊗X
        observable = SparsePauliOp.from_list([("YZX", 1)])

        # Instantiate EstimatorQNN
        estimator = StatevectorEstimator()
        self.qnn = QNN(
            circuit=qc,
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def predict(self, x: list[list[float]]) -> list[float]:
        """Run the quantum circuit for each input in x and return expectation values."""
        return self.qnn.predict(x)

    def fit(self, x: list[list[float]], y: list[float], epochs: int = 10,
            learning_rate: float = 0.01) -> None:
        """Train the weight parameters using gradient descent on the loss."""
        # Simple optimizer loop using qiskit's gradient estimation
        for _ in range(epochs):
            preds = self.predict(x)
            loss = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(x)
            grads = self.qnn.gradient(x, y, weight_params=self.weight_params)
            # Update weights
            for param, grad in zip(self.weight_params, grads):
                param.set_value(param.value() - learning_rate * grad)

__all__ = ["EstimatorQNN"]
