"""Quantum regression model with a multi‑layer variational circuit and hybrid post‑processing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Iterable, Tuple


def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of superposition states and corresponding targets.

    Parameters
    ----------
    num_wires : int
        Number of qubits per sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        States of shape ``(samples, 2**num_wires)`` and targets ``(samples,)``.
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


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for quantum regression, mirroring the classical counterpart.
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


class QModel(tq.QuantumModule):
    """
    Variational quantum circuit with layered ansatz and hybrid classical head.
    """

    class QLayer(tq.QuantumModule):
        """
        Single variational layer consisting of a random unitary followed by
        trainable single‑qubit rotations.
        """

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random layer generates a fixed random unitary per forward pass
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Trainable rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.n_layers = n_layers

        # Encoder that maps classical data to quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Build a stack of variational layers
        self.layers = nn.ModuleList([self.QLayer(num_wires) for _ in range(n_layers)])

        # Entanglement block (optional)
        self.entangle = tq.CNOT(wires=[(i, (i + 1) % num_wires) for i in range(num_wires)])

        # Measurement of Pauli‑X and Pauli‑Y expectation values
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.measure_y = tq.MeasureAll(tq.PauliY)

        # Classical post‑processing head
        self.head = nn.Sequential(
            nn.Linear(2 * num_wires, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid quantum‑classical network.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of classical inputs of shape ``(batch, num_wires)``.

        Returns
        -------
        torch.Tensor
            Predicted regression values of shape ``(batch,)``.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode classical data
        self.encoder(qdev, state_batch)

        # Variational layers
        for layer in self.layers:
            layer(qdev)
            self.entangle(qdev)

        # Expectation values
        exp_x = self.measure_x(qdev)
        exp_y = self.measure_y(qdev)

        # Concatenate features
        features = torch.cat([exp_x, exp_y], dim=1)

        return self.head(features).squeeze(-1)

    # Training utilities ----------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict],
        val_loader: Iterable[dict] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Simple training loop for the hybrid model.

        Parameters
        ----------
        train_loader : Iterable[dict]
            DataLoader yielding batches of ``{states, target}``.
        val_loader : Iterable[dict] | None
            Optional validation DataLoader.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        device : torch.device | str
            Device to run on.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self.forward(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch["states"].to(device)
                        target = batch["target"].to(device)
                        pred = self.forward(states)
                        val_loss += criterion(pred, target).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch:02d} | Train loss: {avg_loss:.4f} | Val loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:02d} | Train loss: {avg_loss:.4f}")

    def predict(self, loader: Iterable[dict], device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Predict on a dataset.

        Parameters
        ----------
        loader : Iterable[dict]
            DataLoader yielding batches.
        device : torch.device | str
            Device to run on.

        Returns
        -------
        torch.Tensor
            Concatenated predictions.
        """
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(device)
                preds.append(self.forward(states))
        return torch.cat(preds, dim=0)

    # Export utilities ------------------------------------------------------
    def to_qiskit_circuit(self, state: np.ndarray) -> tq.QuantumCircuit:
        """
        Convert a single input state into a Qiskit circuit that reproduces the
        same variational parameters.  Useful for benchmarking on real hardware.

        Parameters
        ----------
        state : np.ndarray
            Classical input of shape ``(num_wires,)``.

        Returns
        -------
        tq.QuantumCircuit
            Qiskit circuit representation.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device="cpu")
        self.encoder(qdev, torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        for layer in self.layers:
            layer(qdev)
            self.entangle(qdev)
        return qdev.circuit

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
