import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """
    Variational quantum filter that processes 2×2 image patches.
    Each patch is encoded into 4 qubits via Ry rotations proportional to pixel values.
    A 2‑layer parameterized ansatz (Ry, Rz, CNOT) is applied, and Pauli‑Z expectations are measured.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Parameterized rotation angles: shape (n_layers, n_wires, 2) for Ry and Rz
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 2))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, 28, 28) or (B, 1, 28, 28)
        Returns: Tensor of shape (B, 4*14*14)
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        bsz, H, W = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        patches = []
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = x[:, r:r+2, c:c+2]  # shape (B, 2, 2)
                # Flatten patch to 4 values
                values = patch.view(bsz, -1)  # (B, 4)
                # Encode pixel values into qubits via Ry rotations
                for i in range(self.n_wires):
                    qdev.apply(tq.RY, wires=i, params=values[:, i])
                # Variational ansatz
                for l in range(self.n_layers):
                    for i in range(self.n_wires):
                        qdev.apply(tq.RY, wires=i, params=self.params[l, i, 0])
                        qdev.apply(tq.RZ, wires=i, params=self.params[l, i, 1])
                    # Entangling CNOT chain
                    for i in range(self.n_wires - 1):
                        qdev.apply(tq.CNOT, wires=[i, i+1])
                measurement = self.measure(qdev).view(bsz, self.n_wires)
                patches.append(measurement)
        # Concatenate all patches: shape (B, 4*14*14)
        return torch.cat(patches, dim=1)

class QuanvolutionHybrid(nn.Module):
    """
    Hybrid quantum‑classical classifier.
    The quantum filter extracts 4‑dimensional features per 2×2 patch.
    A residual two‑layer linear head aggregates these features into logits.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_wires, n_layers)
        # Residual linear head
        self.fc1 = nn.Linear(4 * 14 * 14, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        out = self.fc1(features)
        out = self.bn1(out)
        out = F.relu(out)
        residual = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = out + residual
        logits = self.fc3(out)
        return F.log_softmax(logits, dim=-1)
