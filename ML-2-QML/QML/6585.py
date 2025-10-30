import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EstimatorQNN:
    """
    Variational quantum circuit that mirrors the classical EstimatorQNN.

    The circuit operates on 2 qubits.  Classical inputs are encoded as
    rotation angles on the first qubit; weight parameters control the
    entangling layer and a final rotation on the second qubit.  The
    observable is Pauli‑Z on qubit 0, yielding a single real‑valued
    output that can be used for regression.

    The class exposes a `predict` method that evaluates the quantum
    expectation value for a batch of inputs.
    """
    def __init__(self, device: str = "default.qubit", shots: int = 1024):
        self.device = qml.device(device, wires=2, shots=shots)
        # Parameters: first element is input angle, second is weight
        self.input_params = [qml.Param("x")]
        self.weight_params = [qml.Param("w")]

        @self.device
        def circuit(x, w):
            # Input encoding on qubit 0
            qml.RX(x, wires=0)
            qml.RY(x, wires=0)
            # Variational entangling block
            qml.CNOT(wires=[0, 1])
            qml.RZ(w, wires=1)
            qml.CNOT(wires=[0, 1])
            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of 2‑dimensional inputs.

        Parameters
        ----------
        X : array of shape (n_samples, 2)
            Input features; only the first column is used for rotation.
        weights : array of shape (n_samples, 1)
            Weight parameters for the variational layer.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Quantum expectation values.
        """
        preds = []
        for x, w in zip(X, weights):
            preds.append(self.circuit(x[0], w[0]))
        return np.array(preds)

    def __repr__(self) -> str:
        return f"<EstimatorQNN on {self.device.name} with {self.device.shots} shots>"
