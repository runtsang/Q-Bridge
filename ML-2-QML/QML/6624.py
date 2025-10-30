import pennylane as qml
import numpy as np
import torch
from torch.optim import Adam
from typing import Iterable, Tuple

class EstimatorQNN:
    """
    Hybrid quantum‑classical regression model using Pennylane.
    Supports configurable qubits, layers, and observables.
    Trains via gradient descent on the circuit expectation value.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        observables: Iterable = None,
        device_name: str = "default.qubit",
        dev_kwargs: dict | None = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits, **(dev_kwargs or {}))
        self.observables = (
            observables
            or [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        )
        # Random initial parameters
        self.params = np.random.randn(n_layers, n_qubits, 3)  # RX,RZ,RY per layer

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs as RX rotations
            for i, w in enumerate(inputs):
                qml.RX(w, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                    qml.RY(weights[layer, qubit, 2], wires=qubit)
                # Entanglement pattern (ring)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Sum of observables as output
            return sum(qml.expval(obs) for obs in self.observables)

        self.circuit = circuit
        self.weight_params = torch.tensor(self.params, dtype=torch.float32, requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.circuit(inputs, self.weight_params)

    def train(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        opt = Adam([self.weight_params], lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = self.forward(xb)
                loss = torch.mean((pred - yb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f}")

    def evaluate(
        self,
        data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        device: str = "cpu",
    ) -> dict:
        with torch.no_grad():
            preds, targets = [], []
            for xb, yb in data_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds.append(self.forward(xb).cpu())
                targets.append(yb.cpu())
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            mse = torch.mean((preds - targets) ** 2).item()
            mae = torch.mean(torch.abs(preds - targets)).item()
        return {"mse": mse, "mae": mae}

__all__ = ["EstimatorQNN"]
