"""Quantum hybrid network with a quantum quanvolution filter and a quantum expectation head for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter applying a random two‑qubit kernel on 2×2 patches."""
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
        # Assume MNIST‑style 28×28 input; adapt as needed
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
        return torch.cat(patches, dim=1)  # [batch, 4*14*14]


class QuantumHybridHead(tq.QuantumModule):
    """Quantum head that maps a scalar to a probability via a parameterised circuit."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "ry", "wires": [0]}]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch]
        qdev = tq.QuantumDevice(self.random_layer.n_wires, bsz=x.size(0), device=x.device)
        self.encoder(qdev, x.unsqueeze(-1))
        self.random_layer(qdev)
        measurement = self.measure(qdev)
        expectation = measurement[:, 0]  # first qubit
        return expectation.unsqueeze(-1)


class ConvBranch(nn.Module):
    """Convolutional backbone with a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_head = QuantumHybridHead(n_qubits=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [batch, 1]
        x = x.squeeze(-1)  # [batch]
        q_out = self.quantum_head(x)  # [batch, 1]
        probs = torch.sigmoid(q_out)
        return probs


class QuanvolutionBranch(nn.Module):
    """Quantum quanvolution filter branch producing a probability."""
    def __init__(self) -> None:
        super().__init__()
        self.quanvolution = QuantumQuanvolutionFilter()
        # 28×28 input → 14×14 patches → 4*14*14 = 784 features
        self.linear = nn.Linear(784, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)  # [batch, 1, H, W]
        # Resize to 28×28 if necessary
        if gray.shape[2]!= 28 or gray.shape[3]!= 28:
            gray = F.interpolate(gray, size=(28, 28), mode="bilinear", align_corners=False)
        features = self.quanvolution(gray)
        logits = self.linear(features)
        probs = torch.sigmoid(logits)
        return probs


class HybridQuanvolutionNet(nn.Module):
    """Quantum hybrid network combining conv and quanvolution branches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv_branch = ConvBranch()
        self.quanvolution_branch = QuanvolutionBranch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs_conv = self.conv_branch(x)  # [batch, 1]
        probs_q = self.quanvolution_branch(x)  # [batch, 1]
        probs = (probs_conv + probs_q) / 2
        return torch.cat((probs, 1 - probs), dim=-1)
