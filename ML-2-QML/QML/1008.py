"""Quantum regression model with a noise‑augmented variational circuit and a hybrid training hook.

Key additions:
* `NoiseLayer` injects depolarizing noise after each parameterised gate.
* `HybridQModel` exposes a `train_head_only` flag to freeze the encoder during training.
* The circuit now supports a configurable number of random layers for richer feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
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

# --------------------------------------------------------------------------- #
# Noise layer
# --------------------------------------------------------------------------- #
class NoiseLayer(tq.QuantumModule):
    """Depolarizing noise applied after each trainable gate."""
    def __init__(self, p: float = 0.01):
        super().__init__()
        self.p = p

    def forward(self, qdev: tq.QuantumDevice):
        for wire in range(qdev.n_wires):
            qdev.depolarizing(p=self.p, wires=wire)

# --------------------------------------------------------------------------- #
# Variational circuit
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Parameterized layer with optional noise."""
    def __init__(self, n_wires: int, n_ops: int = 30, noise_p: float = 0.01):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.noise = NoiseLayer(p=noise_p)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
            self.noise(qdev)

# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class Encoder(tq.QuantumModule):
    """General encoder using a pre‑defined Ry‑based circuit."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

    def forward(self, qdev: tq.QuantumDevice, states: torch.Tensor):
        self.encoder(qdev, states)

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridQModel(tq.QuantumModule):
    """
    Quantum encoder + classical head. The `train_head_only` flag freezes the encoder
    during training, allowing the head to be fine‑tuned while keeping the quantum
    circuit fixed.
    """
    def __init__(self, num_wires: int, train_head_only: bool = False):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = Encoder(num_wires)
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.train_head_only = train_head_only
        if self.train_head_only:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.q_layer.parameters():
                param.requires_grad = False

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Compatibility wrapper
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """
    Thin wrapper to keep the original API. Internally uses HybridQModel
    with the encoder frozen (default) to match the seed's behaviour.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.net = HybridQModel(num_wires, train_head_only=True)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch)

__all__ = [
    "RegressionDataset",
    "HybridQModel",
    "QModel",
    "generate_superposition_data",
    "QLayer",
    "Encoder",
    "NoiseLayer",
]
