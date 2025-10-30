import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    zero_state = np.zeros(2 ** num_wires, dtype=complex)
    zero_state[0] = 1.0
    one_state = np.zeros(2 ** num_wires, dtype=complex)
    one_state[-1] = 1.0

    thetas = np.random.uniform(0, np.pi, size=samples)
    phis = np.random.uniform(0, 2 * np.pi, size=samples)
    states = np.cos(thetas[:, None]) * zero_state + np.exp(1j * phis[:, None]) * np.sin(thetas[:, None]) * one_state
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

def generate_classification_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(samples, num_features).astype(np.float32)
    y = ((np.sin(X @ np.random.randn(num_features, 1)).sum(axis=1) > 0).astype(np.float32))
    return X, y

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassificationDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumFeatureExtractor(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=25, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

class QModel(nn.Module):
    def __init__(self, num_features: int, use_quantum: bool = False, task: str = "regression"):
        super().__init__()
        self.task = task
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.quantum = QuantumFeatureExtractor(num_features)
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_features}xRy"])
        else:
            self.quantum = None
            self.encoder = None

        self.hidden = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        if self.task == "classification":
            self.head = nn.Linear(32, 2)
        else:
            self.head = nn.Linear(32, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        if self.use_quantum:
            states = batch["states"]
            bsz = states.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.quantum.n_wires, bsz=bsz, device=states.device)
            self.encoder(qdev, states)
            self.quantum(qdev)
            features = tq.MeasureAll(tq.PauliZ)(qdev)
            x = features
        else:
            x = batch["features"]
        x = self.hidden(x)
        out = self.head(x).squeeze(-1)
        return out

__all__ = ["QModel", "RegressionDataset", "ClassificationDataset", "generate_superposition_data", "generate_classification_data"]
