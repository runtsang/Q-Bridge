import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torch.utils.data import DataLoader

class QuanvolutionFilter(tq.QuantumModule):
    """
    Apply a parameterized two-qubit quantum kernel to 2x2 image patches.
    Uses a variational circuit with trainable parameters.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layers = nn.ModuleList([
            tq.RandomLayer(n_ops=4, wires=list(range(n_wires))) for _ in range(n_layers)
        ])
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
                for layer in self.var_layers:
                    layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network using the quanvolutional filter followed by a quantum neural network head.
    Provides fit and predict helpers.
    """
    def __init__(self, num_classes: int = 10, qhead_layers: int = 2):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qhead = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.qhead(features)
        return F.log_softmax(logits, dim=-1)
    def fit(self, train_loader: DataLoader, epochs: int = 5, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        self.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp().argmax(dim=-1)
