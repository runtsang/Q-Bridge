"""Quantum implementation of the hybrid fully‑connected layer.

The class mirrors the classical version but uses a
parameterised quantum circuit built with torchquantum.
It re‑uses the data generation routine from the quantum
seed and exposes a ``run`` method that accepts a list of
angles which are bound to the RX/RY gates before measurement.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states used for regression."""
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset exposing quantum states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridFCL(tq.QuantumModule):
    """Quantum hybrid layer with a random feature map followed by
    trainable RX/RY rotations and a classical read‑out head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 2) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of quantum states."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def run(self, thetas: list[float]) -> np.ndarray:
        """Run the circuit with externally supplied rotation angles.
        The angles are broadcast to all RX and RY gates.
        """
        # broadcast to match number of wires
        batched_thetas = [thetas] * self.n_wires
        # set parameters on the trainable gates
        for gate in [self.q_layer.rx, self.q_layer.ry]:
            gate.params = torch.tensor(batched_thetas, dtype=torch.float32, device=self.head.weight.device).view(1, -1)
        # perform a single forward pass
        with torch.no_grad():
            out = self.forward(torch.zeros((1, 2 ** self.n_wires), dtype=torch.cfloat, device=self.head.weight.device))
        return out.detach().cpu().numpy()

def FCL() -> HybridFCL:
    """Convenience factory matching the original API."""
    return HybridFCL()

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data", "FCL"]
