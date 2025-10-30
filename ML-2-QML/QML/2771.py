"""Quantum hybrid regression model using a variational circuit and a quantum LSTM gate.

The implementation extends the quantum regression seed and incorporates
the quantum LSTM logic from the second seed.  The circuit encodes the
input features, applies a trainable variational layer, and finally
measures expectation values that are fed to a classical linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and a sinusoidal label.

    The states are of the form
        cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    and the target is y = sin(2θ) cos(φ).
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
    """Dataset that returns quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegression(tq.QuantumModule):
    """Hybrid quantum regression model.

    The model encodes the input state, applies a variational layer,
    processes the result through a quantum LSTM gate, measures
    expectation values, and finally maps the features to a real
    regression output using a classical linear head.
    """
    class QLayer(tq.QuantumModule):
        """Small quantum LSTM gate used for feature extraction."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Trainable single‑qubit rotations
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            for gate in self.params:
                gate(qdev)
            # Entangle wires with a ring of CNOTs
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps the classical state to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        features = self.q_layer(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
