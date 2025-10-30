import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import static_support

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel encoder that applies Ry rotations based on input vectors."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

    @static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        # x: [batch, n_wires]
        q_device.reset_states(x.shape[0])
        for i, wire in enumerate(range(self.n_wires)):
            tq.RY(x[:, i], wires=[wire])(q_device)

class QuantumQCNN(tq.QuantumModule):
    """Quantum convolution‑pooling block built from random layers."""
    def __init__(self, n_wires: int, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([tq.RandomLayer(n_ops=20, wires=list(range(n_wires))) for _ in range(depth)])

    @static_support
    def forward(self, q_device: tq.QuantumDevice) -> None:
        for layer in self.layers:
            layer(q_device)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum hybrid model that encodes data, applies QCNN layers, and outputs a scalar."""
    def __init__(self, num_wires: int = 4, conv_depth: int = 3):
        super().__init__()
        self.encoder = KernalAnsatz(num_wires)
        self.qcnn = QuantumQCNN(num_wires, conv_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    @static_support
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.qcnn(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat), "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
