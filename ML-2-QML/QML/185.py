import pennylane as qml
import numpy as np
import torch

class EstimatorQNNGen204QML:
    """
    Variational quantum circuit for regression.

    The circuit uses a two‑qubit ansatz with alternating single‑qubit rotations
    and a CNOT entangling layer. The expectation value of Pauli‑Z on the first
    qubit is used as the output, which is trained to minimise MSE against a
    classical target.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the ansatz. Default 2.
    layers : int
        Number of alternating rotation‑entangle layers. Default 3.
    dev : str | pennylane.Device
        Backend name or device instance. Default "default.qubit".
    """

    def __init__(self,
                 n_qubits: int = 2,
                 layers: int = 3,
                 dev: str | qml.Device = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(dev, wires=n_qubits)
        self.params = np.random.randn(layers * n_qubits * 2) * 0.1
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs with Rx rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                # Entangling CNOT chain
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit on a batch of inputs.

        Parameters
        ----------
        X : np.ndarray
            Input matrix, shape (n_samples, n_qubits).

        Returns
        -------
        np.ndarray
            Circuit outputs, shape (n_samples,).
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        w_t = torch.tensor(self.params, dtype=torch.float32, requires_grad=True)
        preds = []
        for x in X_t:
            preds.append(self.circuit(x, w_t))
        return torch.stack(preds).detach().numpy()

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            verbose: bool = True) -> None:
        """
        Train the variational parameters using Adam and automatic differentiation.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_qubits).
        y : np.ndarray
            Targets, shape (n_samples,).
        epochs : int
            Number of optimisation steps.
        lr : float
            Learning rate.
        verbose : bool
            If True prints loss each 10 epochs.
        """
        w = torch.tensor(self.params, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=lr)
        criterion = torch.nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            preds = self.circuit(X_t, w)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:3d}/{epochs} - Loss: {loss.item():.6f}")
        # Store trained parameters
        self.params = w.detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with the trained circuit.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_qubits).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,).
        """
        return self.forward(X)

__all__ = ["EstimatorQNNGen204QML"]
