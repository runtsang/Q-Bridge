import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states of the form
    cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    Labels are mean and log‑variance derived from theta and phi.
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
    mean = np.sin(2 * thetas) * np.cos(phis)
    logvar = 0.5 + 0.1 * np.sin(thetas)
    labels = np.stack([mean, logvar], axis=1).astype(np.float32)
    return states, labels

class RegressionDataset(Dataset):
    """
    Dataset yielding quantum state tensors and a two‑dimensional target
    (mean, log‑variance) for probabilistic regression.
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

class QProbabilisticModel(tq.QuantumModule):
    """
    Quantum variational circuit that outputs mean and log‑variance.
    Uses a stochastic ensemble of random layers to estimate uncertainty.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=25, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int, ensemble_size: int = 5):
        super().__init__()
        self.n_wires = num_wires
        self.ensemble_size = ensemble_size
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.mean_head = nn.Linear(num_wires, 1)
        self.logvar_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        mean_preds = []
        logvar_preds = []
        for _ in range(self.ensemble_size):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
            self.encoder(qdev, state_batch)
            self.q_layer(qdev)
            features = self.measure(qdev)
            mean_pred = self.mean_head(features).squeeze(-1)
            logvar_pred = self.logvar_head(features).squeeze(-1)
            mean_preds.append(mean_pred)
            logvar_preds.append(logvar_pred)
        mean = torch.stack(mean_preds, dim=0).mean(dim=0)
        logvar = torch.stack(logvar_preds, dim=0).mean(dim=0)
        return mean, logvar

__all__ = ["QProbabilisticModel", "RegressionDataset", "generate_superposition_data"]
