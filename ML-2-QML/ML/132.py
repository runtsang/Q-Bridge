import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from typing import List

class ResidualBlock(nn.Module):
    """A lightweight residual block that adds a skip connection."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class SelfAttention(nn.Module):
    """Optional channel‑wise self‑attention module."""
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.query = nn.Conv1d(dim, dim // reduction, bias=False)
        self.key = nn.Conv1d(dim, dim // reduction, bias=False)
        self.value = nn.Conv1d(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)  # (B, C, N)
        q = self.query(x_flat)     # (B, C//r, N)
        k = self.key(x_flat)       # (B, C//r, N)
        attn = self.softmax(torch.bmm(q.transpose(1, 2), k))  # (B, N, N)
        v = self.value(x_flat)     # (B, C, N)
        out = torch.bmm(v, attn)   # (B, C, N)
        return out.view(b, c, h, w)

class QuantumKernel(nn.Module):
    """Trainable variational kernel that maps 2x2 patches to 4‑dimensional features."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Parameter matrix per layer (theta)
        self.theta = nn.Parameter(torch.randn(n_layers, n_wires, 2))
        # Rotation angles for each qubit
        self.rz = nn.RZ
        self.cnot = nn.CNOT

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch shape: (B, 4)  -> (B, 2, 2) flattened
        B = patch.shape[0]
        state = torch.zeros(B, 2 ** self.n_wires, device=patch.device)
        state[:, 0] = 1.0
        for l in range(self.n_layers):
            for w in range(self.n_wires):
                theta = self.theta[l, w]
                # Apply Ry rotation
                state = self.apply_rz(state, w, theta[0])
                state = self.apply_rz(state, w, theta[1])
            # Entangling layer
            for w in range(self.n_wires - 1):
                state = self.apply_cnot(state, w, w + 1)
        # Measurement in computational basis
        probs = state.abs() ** 2
        return probs[:, :4]  # return first 4 amplitudes as features

    @staticmethod
    def apply_rz(state: torch.Tensor, wire: int, angle: torch.Tensor) -> torch.Tensor:
        # Construct Rz gate matrix
        rz = torch.tensor([[torch.cos(angle / 2), -1j * torch.sin(angle / 2)],
                           [1j * torch.sin(angle / 2), torch.cos(angle / 2)]],
                          device=state.device, dtype=state.dtype)
        # Expand to full dimension
        dim = state.shape[1]
        full = torch.eye(dim, device=state.device, dtype=state.dtype)
        for i in range(dim):
            if (i >> wire) & 1:
                full[i, i] = rz[1, 1]
            else:
                full[i, i] = rz[0, 0]
        return torch.matmul(state, full)

    @staticmethod
    def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        dim = state.shape[1]
        cnot = torch.eye(dim, device=state.device, dtype=state.dtype)
        for i in range(dim):
            if ((i >> control) & 1) and not ((i >> target) & 1):
                j = i ^ (1 << target)
                cnot[i, i] = 0
                cnot[j, j] = 0
                cnot[i, j] = 1
                cnot[j, i] = 1
        return torch.matmul(state, cnot)

class QuanvolutionFilter(nn.Module):
    """Hybrid filter that applies a trainable quantum kernel to each 2x2 patch."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.kernel = QuantumKernel(n_wires, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 1, "Input must be single‑channel grayscale."
        # Extract 2x2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(B, 14 * 14, 4)  # (B, N, 4)
        # Process each patch through the quantum kernel
        features = []
        for i in range(patches.shape[1]):
            feat = self.kernel(patches[:, i, :])  # (B, 4)
            features.append(feat)
        return torch.cat(features, dim=1)  # (B, 4 * 14 * 14)

class QuanvolutionClassifier(nn.Module):
    """Full hybrid model combining classical residual blocks, optional attention, and a quantum filter."""
    def __init__(self, use_attention: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(1)
        self.res2 = ResidualBlock(1)
        self.attn = SelfAttention(1) if use_attention else None
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, 10)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # fusion weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        cls_feat = self.res1(x)
        cls_feat = self.res2(cls_feat)
        if self.attn:
            cls_feat = self.attn(cls_feat)
        cls_feat = cls_feat.view(x.size(0), -1)
        # Quantum path
        q_feat = self.qfilter(x)
        # Fuse
        fused = self.alpha * cls_feat + (1 - self.alpha) * q_feat
        logits = self.fc(fused)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
