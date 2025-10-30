import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

class RegressionDataset(tq.utils.Dataset):
    """Dataset generating quantum states and labels for regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate_data(num_wires, samples)

    def _generate_data(self, num_wires: int, samples: int):
        omega0 = np.zeros(2**num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2**num_wires, dtype=complex)
        omega1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.cos(thetas)[:, None] * omega0 + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * omega1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states.astype(np.complex64), labels.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class HybridQuantumLayer(tq.QuantumModule):
    """Variational layer with random gates and trainable rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

class QuantumRegressionHybrid(tq.QuantumModule):
    """Hybrid model: classical encoding + quantum variational layer + linear head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.quantum_layer = HybridQuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.quantum_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionDataset", "QuantumRegressionHybrid", "HybridQuantumLayer"]
