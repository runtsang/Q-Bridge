import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torch.utils.data import Dataset

class HybridRegressionDataset(Dataset):
    """Quantum dataset providing states and targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate_states(samples, num_wires)

    def _generate_states(self, samples: int, num_wires: int):
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

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model with encoder, variational layer, sampler, and linear head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        # Sampler circuit inspired by SamplerQNN
        self.sampler_layer = tq.RandomLayer(n_ops=10, wires=list(range(num_wires)))

    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def forward(self, batch: dict) -> torch.Tensor:
        bsz = batch["states"].shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=batch["states"].device)
        self.encoder(qdev, batch["states"])
        self.q_layer(qdev)
        self.sampler_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """Predict directly from raw quantum states."""
        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=states.device)
        self.encoder(qdev, states)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionDataset", "HybridRegressionModel"]
