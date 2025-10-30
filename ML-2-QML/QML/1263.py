"""Quantum implementation of EstimatorQNNGen209.

This module uses PennyLane to construct a variational circuit that takes a
single real input, encodes it via an RY rotation, applies a two‑qubit
entangling block, and measures the expectation of a Pauli‑Z operator.
The class exposes a simple fit/predict API that mirrors the classical
counterpart.
"""

import pennylane as qml
import torch
from torch import nn, optim

class EstimatorQNNGen209:
    """
    A variational quantum regressor implemented with PennyLane.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the circuit.
    n_layers : int, default 2
        Number of variational layers.
    device_name : str, default "default.qubit"
        PennyLane device for simulation.
    """
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits)

        # Parameter shape: (n_layers, n_qubits, 3) for RX,RZ,RY rotations
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )

        # Wrap the circuit in a QNode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(x, params):
            # Input encoding: one RY gate per qubit
            for w in range(self.n_qubits):
                qml.RY(x[w], wires=w)

            # Variational layers
            for layer in range(self.n_layers):
                for w in range(self.n_qubits):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RZ(params[layer, w, 1], wires=w)
                    qml.RY(params[layer, w, 2], wires=w)

                # Entangling CNOT chain
                for w in range(self.n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])

            # Measurement: expectation of PauliZ on all qubits, summed
            return sum(qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits))

        self.circuit = circuit

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        x : torch.Tensor of shape (n_qubits,)
            Input features encoded as rotation angles.
        """
        return self.circuit(x, self.params)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> list[float]:
        """
        Train the quantum parameters using Adam optimizer.

        Parameters
        ----------
        X : torch.Tensor of shape (N, n_qubits)
            Training inputs.
        y : torch.Tensor of shape (N, 1)
            Target values.
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = optim.Adam([self.params], lr=lr)
        criterion = nn.MSELoss()

        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb, yb  # keep tensors on CPU
                opt.zero_grad()
                preds = []
                for i in range(xb.size(0)):
                    preds.append(self.circuit(xb[i], self.params))
                preds = torch.stack(preds)
                loss = criterion(preds, yb.squeeze())
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            history.append(epoch_loss / len(loader.dataset))

        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return predictions for a batch of inputs."""
        preds = []
        for xb in X:
            preds.append(self.circuit(xb, self.params).detach())
        return torch.stack(preds)

def EstimatorQNN() -> EstimatorQNNGen209:
    """Return an instance of the upgraded EstimatorQNNGen209."""
    return EstimatorQNNGen209()
