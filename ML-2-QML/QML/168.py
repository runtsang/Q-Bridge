"""Quantum regression model built with Pennylane.

The module defines a parameterised variational circuit that accepts batches of
classical state vectors, encodes them via a custom layer, applies an
entangling block, and reads out expectation values of Pauli‑Z operators.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable, Optional, Tuple


def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum regression data.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray
        State vectors of shape (samples, 2**num_wires), dtype=complex.
    labels : np.ndarray
        Scalar targets of shape (samples,).
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the quantum regression data.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Quantum‑classical hybrid regression model using Pennylane.

    Parameters
    ----------
    num_wires : int
        Number of qubits used in the variational circuit.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires

        # Device for the variational circuit
        self.dev = qml.device("default.qubit", wires=num_wires)

        # Trainable parameters of the variational block
        self.params = nn.Parameter(torch.randn(num_wires))

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # Build the variational circuit
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self):
        """
        Return a Pennylane qnode that implements a simple entangling block.
        """

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # State preparation: map each amplitude to a rotation around Z
            for i in range(self.num_wires):
                qml.RZ(state[i], wires=i)

            # Entangling block with trainable RY rotations
            for i in range(self.num_wires):
                qml.CNOT(wires=[i, (i + 1) % self.num_wires])
                qml.RY(params[i], wires=i)

            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

        return circuit

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of state vectors of shape (batch_size, 2**num_wires).
        """
        batch_size = state_batch.shape[0]
        # Convert complex state vectors into real phases for RZ rotations
        phases = torch.angle(state_batch)  # shape (batch_size, 2**num_wires)
        # Use the first num_wires phases as inputs
        input_phases = phases[:, :self.num_wires]

        # Map the circuit over the batch
        outputs = qml.map(
            lambda ph, par: self.circuit(ph, par),
            input_phases,
            self.params,
            batch_size=batch_size,
        )

        # Stack outputs into a tensor of shape (batch_size, num_wires)
        features = torch.stack(outputs, dim=0)
        return self.head(features).squeeze(-1)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        dataloader: Iterable[dict],
        epochs: int = 10,
        lr: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> list[float]:
        """
        Train the hybrid model using MSE loss and Adam optimizer.

        Parameters
        ----------
        dataloader : Iterable[dict]
            Iterable yielding batches of ``{'states': Tensor, 'target': Tensor}``.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        device : torch.device, optional
            Device to run on; defaults to CUDA if available.

        Returns
        -------
        losses : list[float]
            Training loss after each epoch.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states, target = batch["states"].to(device), batch["target"].to(device)
                optimizer.zero_grad()
                pred = self(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            losses.append(epoch_loss / len(dataloader.dataset))
        return losses

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return predictions for a batch of state vectors.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(X)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
