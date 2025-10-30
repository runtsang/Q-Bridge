import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class EstimatorQNN:
    """
    Qiskit implementation of a variational quantum neural network that
    accepts *n* input qubits, encodes each feature with a Ry rotation,
    applies a trainable Rz layer, and measures the first qubit in the Z basis.
    The class provides a lightweight training loop that uses the
    parameter‑shift rule to compute analytic gradients for the weight
    parameters.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 1, device: str = "statevector"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device

        # Parameter lists
        self.input_params = [Parameter(f"x{i}") for i in range(num_qubits)]
        self.weight_params = [Parameter(f"w{i}") for i in range(num_qubits)]

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rz(self.weight_params[i], i)

        # Entangling layers
        for _ in range(depth):
            for i in range(num_qubits - 1):
                self.circuit.cx(i, i + 1)
            self.circuit.cx(num_qubits - 1, 0)

        # Observable: measure Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator primitive
        self.estimator = QiskitEstimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN on a batch of inputs.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = []
        for sample in X:
            # Bind input parameters; keep weight_params as trainable
            bindings = {p: v for p, v in zip(self.input_params, sample)}
            result = self.estimator.run(
                circuits=[self.circuit],
                parameter_bindings=[bindings],
                observables=[self.observable],
            ).result()
            predictions.append(result.values[0])
        return np.array(predictions)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.01):
        """
        Train the weight parameters using the parameter‑shift rule.
        """
        # Convert weights to numpy for easy manipulation
        weights = np.zeros(self.num_qubits)
        for epoch in range(epochs):
            loss = 0.0
            grads = np.zeros_like(weights)
            for xi, yi in zip(X, y):
                # Forward pass
                preds = self.predict(xi.reshape(1, -1))[0]
                loss += (preds - yi) ** 2

                # Parameter‑shift gradients
                for i in range(self.num_qubits):
                    shift = np.pi / 2
                    # + shift
                    bindings_plus = {p: v for p, v in zip(self.input_params, xi)}
                    bindings_plus[self.weight_params[i]] = weights[i] + shift
                    plus = self.estimator.run(
                        circuits=[self.circuit],
                        parameter_bindings=[bindings_plus],
                        observables=[self.observable],
                    ).result().values[0]
                    # - shift
                    bindings_minus = {p: v for p, v in zip(self.input_params, xi)}
                    bindings_minus[self.weight_params[i]] = weights[i] - shift
                    minus = self.estimator.run(
                        circuits=[self.circuit],
                        parameter_bindings=[bindings_minus],
                        observables=[self.observable],
                    ).result().values[0]
                    grad = (plus - minus) / 2
                    grads[i] += 2 * (preds - yi) * grad  # chain rule

            loss /= len(X)
            grads /= len(X)
            # Gradient descent step
            weights -= lr * grads

            # Update the weight parameters in the circuit
            for i, w in enumerate(weights):
                self.circuit.assign_parameters({self.weight_params[i]: w}, inplace=True)

            print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.6f}")
