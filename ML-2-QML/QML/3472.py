import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torchquantum import QuantumDevice

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states and corresponding labels."""
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

class RegressionDatasetQuantum(torch.utils.data.Dataset):
    """Quantum regression dataset mirroring the classical counterpart."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QSamplerModule(tq.QuantumModule):
    """Quantum sampler that emits a probability distribution over two basis states."""
    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rx = tq.RX(has_params=True, trainable=True)
        self.cx = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice):
        self.ry(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.cx(qdev, control=0, target=1)
        self.rx(qdev, wires=0)
        self.rx(qdev, wires=1)
        self.cx(qdev, control=0, target=1)
        self.ry(qdev, wires=0)
        self.ry(qdev, wires=1)

    def sample(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(n_wires=2, bsz=bsz, device=state_batch.device)
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2xRy"])
        encoder(qdev, state_batch)
        self.forward(qdev)
        probs = self.measure(qdev).squeeze(-1)
        return probs

class QModel(tq.QuantumModule):
    """Variational quantum regression head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
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
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev).squeeze(-1)
        return self.head(features).squeeze(-1)

class UnifiedSamplerRegressorQuantum(tq.QuantumModule):
    """Hybrid quantum‑classical model that outputs both a probability distribution and a regression value."""
    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.sampler = QSamplerModule(n_wires=num_wires)
        self.regressor = QModel(num_wires=num_wires)

    def forward(self, state_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = self.sampler.sample(state_batch)
        target = self.regressor(state_batch)
        return {"probs": probs, "target": target}

__all__ = [
    "generate_superposition_data",
    "RegressionDatasetQuantum",
    "QSamplerModule",
    "QModel",
    "UnifiedSamplerRegressorQuantum",
]
