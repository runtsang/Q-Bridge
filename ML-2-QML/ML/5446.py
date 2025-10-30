import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic superposition data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridEstimatorQNN(nn.Module):
    """Hybrid classical‑quantum regressor.

    The model first embeds the raw features into a low‑dimensional
    classical space, then feeds the embedding into a variational
    quantum circuit implemented with TorchQuantum.  The quantum
    layer acts as a learnable non‑linear feature map that is
    followed by a linear head for regression.
    """

    def __init__(self, num_features: int = 4, n_wires: int = 4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_wires),
            nn.ReLU(),
        )
        self.quantum_layer = self._build_q_layer(n_wires)
        self.head = nn.Linear(n_wires, 1)

    def _build_q_layer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)

        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        bsz = features.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.quantum_layer.n_wires, bsz=bsz, device=features.device)
        self.quantum_layer(qdev)
        q_features = tq.MeasureAll(tq.PauliZ)(qdev)
        return self.head(q_features).squeeze(-1)

__all__ = ["HybridEstimatorQNN", "RegressionDataset", "generate_superposition_data"]
