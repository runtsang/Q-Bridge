"""
EstimatorQNN (quantum) – a hybrid variational circuit with classical read‑out, trainable via PennyLane.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Iterable, Tuple, Optional


class EstimatorQNN(nn.Module):
    """
    A parameter‑efficient variational quantum circuit that estimates a scalar
    target from two classical inputs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the variational circuit.
    hidden_layers : Iterable[int], optional
        Number of rotation gates per layer. Defaults to (2,).
    device : str, optional
        PennyLane device name (e.g. ``"default.qubit"``).
    """

    def __init__(
        self,
        num_qubits: int = 1,
        hidden_layers: Iterable[int] | None = None,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers) if hidden_layers is not None else [2]
        self.num_qubits = num_qubits
        self.hidden_layers = hidden_layers

        # Create a PennyLane device
        self.dev = qml.device(device, wires=num_qubits)

        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode two classical inputs using Ry rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=0)

            # Parameterised layers
            idx = 0
            for layer in hidden_layers:
                for _ in range(layer):
                    qml.RX(weights[idx], wires=0)
                    qml.RZ(weights[idx + 1], wires=0)
                    idx += 2

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Initialize trainable weights
        num_params = sum(hidden_layers) * 2
        self.weight_params = nn.Parameter(torch.randn(num_params))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Evaluate the variational circuit on a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch_size, 2) containing two classical features.
        """
        batch_size = inputs.shape[0]
        outputs = torch.stack([self.circuit(inp, self.weight_params) for inp in inputs])
        return outputs.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def train_on_dataset(
        self,
        dataset: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stop_patience: int | None = None,
        device: str = "cpu",
    ) -> Tuple[list[float], list[float]]:
        """
        Train the hybrid model on the supplied dataset.

        Returns
        -------
        train_losses, val_losses
        """
        self.to(device)
        inputs, targets = dataset
        dataset = torch.utils.data.TensorDataset(inputs.to(device), targets.to(device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses: list[float] = []
        val_losses: list[float] = []

        best_val = float("inf")
        wait = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            train_losses.append(epoch_loss)

            # Validation on the same data
            self.eval()
            with torch.no_grad():
                val_pred = self(inputs.to(device))
                val_loss = criterion(val_pred, targets.to(device)).item()
                val_losses.append(val_loss)

            if early_stop_patience is not None and val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1

            if early_stop_patience is not None and wait >= early_stop_patience:
                break

        return train_losses, val_losses


__all__ = ["EstimatorQNN"]
