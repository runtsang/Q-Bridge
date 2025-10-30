"""Quantum regression model using PennyLane with a variational circuit.

The module re‑implements the original quantum seed but extends it with:
- configurable number of hidden layers in the ansatz,
- entangling layers (CNOT) for richer expressivity,
- a classical post‑processing head,
- a lightweight training routine that can run on any Pennylane backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic quantum regression dataset where the target depends
    on the superposition angle and phase.

    Parameters
    ----------
    num_wires : int
        Number of qubits used to encode the input state.
    samples : int
        Number of data points.

    Returns
    -------
    states, labels : np.ndarray
        `states` shape (samples, 2**num_wires) complex amplitudes,
        `labels` shape (samples,) real targets.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the quantum‑state data.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the circuit.
    hidden_layers : int, optional
        Number of variational layers after the encoding.
    entangle : bool, optional
        If True, adds a CNOT layer between each pair of qubits in every variational layer.
    backend : str, optional
        PennyLane backend name (default 'default.qubit').
    """
    def __init__(
        self,
        num_wires: int,
        hidden_layers: int = 2,
        entangle: bool = True,
        backend: str = "default.qubit",
        device: str | None = None,
    ):
        super().__init__()
        self.num_wires = num_wires
        self.hidden_layers = hidden_layers
        self.entangle = entangle

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # PennyLane device
        self.dev = qml.device(backend, wires=num_wires, shots=0, device=device)

        # Parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(hidden_layers, num_wires, 3, dtype=torch.float32))

        # Register the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(state, params):
            # State preparation
            qml.QubitStateVector(state, wires=range(num_wires))
            # Variational layers
            for layer in range(hidden_layers):
                for w in range(num_wires):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RY(params[layer, w, 1], wires=w)
                    qml.RZ(params[layer, w, 2], wires=w)
                if entangle:
                    for w in range(num_wires - 1):
                        qml.CNOT(wires=[w, w + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 2**num_wires) complex amplitudes.

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (batch,)
        """
        bsz = state_batch.shape[0]
        # Run the circuit in parallel
        features = self.circuit(state_batch, self.params)
        # features: (batch, num_wires)
        return self.head(features).squeeze(-1)

    @staticmethod
    def train_loop(
        model: "QModel",
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        lr: float,
        device: torch.device,
        early_stop_patience: int = 10,
    ) -> tuple["QModel", list[float], list[float]]:
        """
        Training routine for the hybrid model that mirrors the classical
        training loop. Uses PennyLane's autograd via the torch interface.
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses, val_losses = [], []
        best_val = float("inf")
        patience = 0
        model.to(device)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                preds = model(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch["states"].size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        preds = model(batch["states"].to(device))
                        loss = criterion(preds, batch["target"].to(device))
                        val_loss += loss.item() * batch["states"].size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    torch.save(model.state_dict(), "best_qmodel.pt")
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                val_losses.append(float("nan"))

        if val_loader is not None:
            model.load_state_dict(torch.load("best_qmodel.pt"))
        return model, train_losses, val_losses

    def evaluate(self, loader: DataLoader, device: torch.device) -> float:
        """
        Compute mean squared error on the given loader.
        """
        self.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                preds = self(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                total_loss += loss.item() * batch["states"].size(0)
        return total_loss / len(loader.dataset)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
