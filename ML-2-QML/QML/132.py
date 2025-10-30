import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np

class VariationalPatchKernel(nn.Module):
    """
    A trainable variational circuit that operates on 2×2 image patches
    and outputs a 4‑dimensional feature vector.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, device: str = "cpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None, device=device)
        self.qfunc = qml.QNode(self._circuit, self.dev, interface="torch")
        # Learnable parameters for each layer
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 2))

    def _circuit(self, patch: np.ndarray, params: np.ndarray):
        """
        patch: shape (4,) – flatten 2×2 pixel values
        params: shape (n_layers, n_qubits, 2)
        """
        # Encode pixel values with Ry rotations
        for w in range(self.n_qubits):
            qml.RY(patch[w], wires=w)
        # Apply variational layers
        for l in range(self.n_layers):
            for w in range(self.n_qubits):
                qml.RY(params[l, w, 0], wires=w)
                qml.RZ(params[l, w, 1], wires=w)
            # Entangling layer
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
        # Measure expectation of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        patch: shape (B, 4)
        Returns: (B, 4) feature vectors
        """
        B = patch.shape[0]
        out = []
        for i in range(B):
            out.append(self.qfunc(patch[i].cpu().numpy(), self.params[i].cpu().numpy()))
        return torch.tensor(out, device=patch.device, dtype=torch.float32)

class QuanvolutionFilterQuantum(nn.Module):
    """
    Quantum‑only filter that applies the VariationalPatchKernel to each
    2×2 patch of a batch of images.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, device: str = "cpu"):
        super().__init__()
        self.kernel = VariationalPatchKernel(n_qubits, n_layers, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 1, "Input must be single‑channel."
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B,1,14,14,2,2)
        patches = patches.contiguous().view(B, 14 * 14, 4)  # (B,N,4)
        feats = []
        for i in range(patches.shape[1]):
            feats.append(self.kernel(patches[:, i, :]))
        return torch.cat(feats, dim=1)  # (B, 4*14*14)

class QuanvolutionClassifierQuantum(nn.Module):
    """
    End‑to‑end hybrid model that fuses a classical convolutional backbone
    with a quantum variational filter.
    """
    def __init__(self, backbone: nn.Module, n_qubits: int = 4, n_layers: int = 3, device: str = "cpu"):
        super().__init__()
        self.backbone = backbone
        self.qfilter = QuanvolutionFilterQuantum(n_qubits, n_layers, device)
        self.fc = nn.Linear(4 * 14 * 14, 10)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable fusion weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical backbone
        cls_feat = self.backbone(x)  # shape (B, C, H, W)
        cls_feat = torch.mean(cls_feat, dim=[2, 3])  # global avg pool
        cls_feat = cls_feat.view(x.size(0), -1)      # (B, C)
        # Quantum path
        q_feat = self.qfilter(x)  # (B, 4*14*14)
        # Fuse
        fused = self.alpha * cls_feat + (1 - self.alpha) * q_feat
        logits = self.fc(fused)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilterQuantum", "QuanvolutionClassifierQuantum"]
