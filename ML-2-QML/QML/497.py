"""Quantum regression model using Pennylane and a hybrid variational circuit.

The module provides an amplitude‑encoded dataset generator, a PyTorch Dataset
wrapper, and a `HybridRegression` class that embeds a PennyLane
parameter‑ised circuit followed by a classical read‑out head.  The circuit
consists of an angle‑encoding layer, a depth‑controlled entanglement block,
and a measurement of Pauli‑Z expectations.  The model is fully differentiable
with respect to both quantum and classical parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml


def generate_superposition_data(num_wires: int, samples: int, *,
                                 seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits/wires in the circuit.
    samples : int
        Number of samples to generate.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    states, labels : np.ndarray
        ``states`` has shape ``(samples, 2**num_wires)`` with complex entries
        describing a superposition |ψ⟩.  ``labels`` is a real‑valued array of
        shape ``(samples,)`` computed from the parameters that created the
        state.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)

    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        # |ψ⟩ = cosθ |0…0⟩ + e^{iφ} sinθ |1…1⟩
        states[i, 0] = np.cos(thetas[i])
        states[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch Dataset wrapping the quantum state data."""

    def __init__(self, samples: int, num_wires: int, *, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed=seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """Hybrid quantum‑classical regression model.

    The model embeds a PennyLane variational circuit that processes an amplitude‑
    encoded state.  The circuit consists of an angle‑encoding layer (RX/RZ),
    a configurable number of entanglement layers (CX gates), and a
    measurement of Pauli‑Z expectations.  The resulting feature vector is
    fed into a classical linear head.
    """

    def __init__(self, num_wires: int, *,
                 entanglement_depth: int = 2,
                 entanglement_type: str = "cx",
                 device: str | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.device = device or "default.qubit"
        self.entanglement_depth = entanglement_depth
        self.entanglement_type = entanglement_type

        # Define a trainable quantum node
        @qml.qnode(
            qml.device(self.device, wires=self.num_wires, shots=0),
            interface="torch",
            diff_method="backprop",
        )
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # Amplitude encoding via a custom basis‑state preparation
            # (here we simply set the amplitudes directly for illustration)
            qml.StatePreparation(state, wires=range(self.num_wires))

            # Parameterised rotation layer
            for i in range(self.num_wires):
                qml.RX(params[i], wires=i)
                qml.RZ(params[i], wires=i)

            # Entanglement block
            for _ in range(self.entanglement_depth):
                if self.entanglement_type == "cx":
                    for i in range(self.num_wires - 1):
                        qml.CX(wires=[i, i + 1])
                else:
                    # Alternative: all‑to‑all CZ
                    for i in range(self.num_wires):
                        for j in range(i + 1, self.num_wires):
                            qml.CZ(wires=[i, j])

            # Measurement of expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        self.circuit = circuit
        # Initialise parameters
        params_shape = (self.num_wires,)
        self.params = nn.Parameter(torch.randn(params_shape, dtype=torch.float32))

        # Classical read‑out head
        self.head = nn.Linear(self.num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch has shape (batch, 2**num_wires) and dtype complex
        batch_size = state_batch.shape[0]
        # Ensure state_batch is torch.cfloat
        state_batch = state_batch.to(torch.cfloat)
        # Compute quantum expectation values
        q_features = self.circuit(state_batch, self.params)  # shape (batch, num_wires)
        # Classical head
        return self.head(q_features).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
