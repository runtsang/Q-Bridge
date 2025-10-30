"""Quantum regression model with a deep variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple, Dict

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Labels are a nonlinear function of theta and phi.
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
    Dataset wrapping the quantum state vectors and labels.
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
    Variational quantum circuit followed by a classical read‑out head.
    """
    class QLayer(tq.QuantumModule):
        """
        A deep block of parameterized single‑qubit rotations followed by
        a layer of entangling CNOT gates. The depth is controlled by ``n_layers``.
        """
        def __init__(self, num_wires: int, n_layers: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.n_layers = n_layers
            self.rotation = tq.RX(has_params=True, trainable=True)
            self.entangle = tq.CNOT()

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                for wire in range(self.num_wires):
                    self.rotation(qdev, wires=wire)
                # entangle adjacent qubits in a ring
                for wire in range(self.num_wires):
                    self.entangle(qdev, wires=[wire, (wire + 1) % self.num_wires])

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        # Encoder that maps a classical vector into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head: linear layer from n_wires expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode the batch of classical states, run the variational circuit,
        and read out the expectation values.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def training_step(self, batch: Dict[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
        preds = self(batch["states"])
        loss = criterion(preds, batch["target"])
        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            preds = self(batch["states"])
            return preds, batch["target"]

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
