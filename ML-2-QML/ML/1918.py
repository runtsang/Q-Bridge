import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Tuple

class ResidualBlock(nn.Module):
    """A simple residual block for 2‑D convolutional feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class AttentionReadout(nn.Module):
    """Learnable attention map over spatial features, projecting to 4 logits."""
    def __init__(self, in_dim: int, out_dim: int=4):
        super().__init__()
        self.attention = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        attn = torch.sigmoid(self.attention(x))          # (B, 1, H, W)
        weighted = x * attn                              # broadcast
        pooled = weighted.mean(dim=[2, 3])               # (B, C)
        logits = self.project(pooled)                    # (B, 4)
        return logits

class QuantumNATExtended(nn.Module):
    """
    Hybrid classical-quantum architecture:
      * Residual CNN backbone (2 layers)
      * Attention readout
      * Optional quantum layer for feature fusion
    """
    def __init__(self, use_quantum: bool=True, n_wires: int=4):
        super().__init__()
        self.backbone = nn.Sequential(
            ResidualBlock(1, 8, stride=1),
            ResidualBlock(8, 16, stride=2)
        )
        self.readout = AttentionReadout(16, 4)
        self.use_quantum = use_quantum
        if use_quantum:
            # simple parameterized circuit acting on 4 qubits
            self.quantum_layer = nn.Linear(16, n_wires)  # placeholder for a real Ansatz
            self.bn = nn.BatchNorm1d(n_wires)
        else:
            self.quantum_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) typical MNIST shape
        features = self.backbone(x)                    # (B, 16, 14, 14)
        logits = self.readout(features)                # (B, 4)
        if self.use_quantum:
            # encode logits as a simple feature vector for the Ansatz
            q_input = self.quantum_layer(logits)       # (B, n_wires)
            out = self.bn(q_input)                    # emulate measurement statistics
            return out
        return logits

    def fit(self, train_loader, criterion, optimizer, epochs=10, device='cpu'):
        self.to(device)
        self.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        self.eval()
