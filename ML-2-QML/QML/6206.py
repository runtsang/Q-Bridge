import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

# --------------------------------------------------------------------------- #
# Quantum feature‑map ansatz with random entanglement and trainable rotations
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Programmable quantum feature map with a random layer and trainable rotations."""
    def __init__(self, n_wires: int, n_ops: int = 30):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device states
        q_device.reset_states(x.shape[0])

        # Encode x with trainable rotations
        for wire in range(self.n_wires):
            self.rx(q_device, wires=wire, params=x[:, wire] if x is not None else None)
            self.ry(q_device, wires=wire, params=x[:, wire] if x is not None else None)

        # Entangle
        self.random_layer(q_device)

        # Encode y in reverse order with negative parameters
        for wire in reversed(range(self.n_wires)):
            self.rx(q_device, wires=wire, params=-y[:, wire] if y is not None else None)
            self.ry(q_device, wires=wire, params=-y[:, wire] if y is not None else None)

# --------------------------------------------------------------------------- #
# Quantum kernel that computes the absolute overlap of two encoded states
# --------------------------------------------------------------------------- #
class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap between two encoded states."""
    def __init__(self, n_wires: int = 4, n_ops: int = 30):
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = KernalAnsatz(n_wires, n_ops)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap between the two encoded states."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix between two sets of feature vectors."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Quantum regression dataset using superposition states
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Quantum regression dataset using superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum regression model with a feature‑extracting layer and a classical head
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """Quantum regression model with a feature‑extracting layer and a classical head."""
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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "RegressionDataset",
    "generate_superposition_data",
    "QModel",
]
