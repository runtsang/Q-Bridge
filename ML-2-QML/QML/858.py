"""Quantum classifier with configurable entanglement and measurement‑based loss.

The module defines a single ``QuantumClassifierModel`` class that builds a
parameter‑efficient ansatz and trains it end‑to‑end with a classical optimiser.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
import torch.optim as optim

class QuantumClassifierModel:
    """Quantum classifier with configurable entanglement and measurement‑based loss.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) used for encoding.
    depth : int, default 2
        Number of variational layers.
    entanglement : str, default 'linear'
        Entanglement pattern: 'linear', 'full', or 'none'.
    device : str, default 'default.qubit'
        Pennylane device name.
    shots : int, default 1024
        Number of shots for expectation estimation.
    learning_rate : float, default 0.01
        Optimiser step size.
    epochs : int, default 100
        Number of training epochs.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        entanglement: str = "linear",
        device: str = "default.qubit",
        shots: int = 1024,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.device = qml.device(device, wires=num_qubits, shots=shots)

        # Parameter vector: one rotation per qubit per layer
        self.params = np.random.randn(depth, num_qubits)

        # Observables: one Z per qubit
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Compile the QNode
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Parameterized circuit that encodes ``x`` and applies variational layers."""
        # Data encoding
        for i in range(self.num_qubits):
            qml.RX(x[i], wires=i)

        # Variational layers
        for layer in range(self.depth):
            for qubit in range(self.num_qubits):
                qml.RY(params[layer, qubit], wires=qubit)
            # Entangling gates
            if self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qml.CZ(wires=[i, j])
            elif self.entanglement == "linear":
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # else: no entanglement

        # Measurement
        return [qml.expval(obs) for obs in self.observables]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return expectation values for input ``x``."""
        return self.qnode(x, torch.tensor(self.params, requires_grad=True))

    def train(self, data_loader, device: torch.device | str = "cpu") -> None:
        """End‑to‑end training loop using Adam and cross‑entropy loss."""
        optimizer = optim.Adam([torch.tensor(self.params, requires_grad=True)], lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = torch.stack([self.forward(x) for x in batch_x])
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(data_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs} – loss: {epoch_loss:.4f}")

    def predict(self, x: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return measurement results for input ``x``."""
        self.qnode.device.set_options(shots=1)  # use deterministic expectation
        with torch.no_grad():
            return torch.stack([self.forward(xi) for xi in x])

__all__ = ["QuantumClassifierModel"]
