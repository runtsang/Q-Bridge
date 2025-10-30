import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and regression labels."""
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
    """Dataset for either 1‑D quantum data or 2‑D image‑based data."""
    def __init__(self, num_wires: int = 4, samples: int = 10000, is_2d: bool = False):
        self.is_2d = is_2d
        if is_2d:
            self.data = torch.randn(samples, 1, 28, 28)
            self.target = self.data.view(samples, -1).sum(dim=1)
        else:
            self.states, self.labels = generate_superposition_data(num_wires, samples)
    def __len__(self):  # type: ignore[override]
        return len(self.data) if hasattr(self, "data") else len(self.states)
    def __getitem__(self, idx: int):  # type: ignore[override]
        if hasattr(self, "data"):
            return {"states": self.data[idx], "target": self.target[idx]}
        else:
            return {"states": self.states[idx], "target": self.labels[idx]}

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum two‑qubit kernel applied to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class _QLayer(tq.QuantumModule):
    """Variational layer used for 1‑D quantum regression."""
    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

class QuantumHybridRegression(tq.QuantumModule):
    """
    Quantum hybrid regression model.
    1‑D mode: variational circuit + linear head.
    2‑D mode: quantum quanvolution + linear head.
    """
    def __init__(self, num_wires: int = 4) -> None:
        super().__init__()
        self.is_2d = False
        self.n_wires = num_wires
        # 1‑D encoder and quantum layer
        self.encoder_1d = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_wires)]
        )
        self.q_layer = _QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        # 2‑D quanvolution
        self.qfilter = QuanvolutionFilter()
        self.head_2d = nn.Linear(4 * 14 * 14, 1)
    def set_2d(self, flag: bool = True) -> None:
        self.is_2d = flag
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.is_2d:
            features = self.qfilter(state_batch)
            return self.head_2d(features).squeeze(-1)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder_1d(qdev, state_batch)
        self.q_layer(qdev)
        feats = self.measure(qdev)
        return self.head(feats).squeeze(-1)

__all__ = ["RegressionDataset", "QuantumHybridRegression", "QuanvolutionFilter"]
