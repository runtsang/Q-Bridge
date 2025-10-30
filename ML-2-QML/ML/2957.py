import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuantumKernel(nn.Module):
    """
    Random Fourier feature approximation of a 2‑qubit quantum kernel.
    Each 2x2 image patch (4 values) is mapped to a 4‑dimensional feature vector.
    """
    def __init__(self, input_dim: int = 4, out_dim: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(input_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (batch, n_patches, 4)
        return torch.sin(torch.einsum("bpf,fo->bpo", patches, self.W) + self.b)

class QuantumHybridNAT(nn.Module):
    """
    Classical‑only hybrid model that combines a CNN backbone,
    a random‑Fourier quantum‑kernel patch encoder, and a linear classification head.
    """
    def __init__(self) -> None:
        super().__init__()
        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_backbone = nn.Linear(16 * 7 * 7, 64)

        # Classical quantum‑kernel encoder
        self.kernel = ClassicalQuantumKernel()
        self.fc_kernel = nn.Linear(4 * 14 * 14, 64)

        # Final classifier
        self.fc_combined = nn.Linear(64 + 64, 10)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split a batch of 28x28 images into 2x2 patches.
        Returns a tensor of shape (batch, 14*14, 4).
        """
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, :, r:r+2, c:c+2].view(bsz, -1)
                patches.append(patch)
        return torch.stack(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN path
        features = self.backbone(x)
        flat = features.view(x.shape[0], -1)
        cnn_out = self.fc_backbone(flat)

        # Quantum‑kernel path
        patches = self._extract_patches(x)
        kernel_out = self.kernel(patches)
        kernel_feat = kernel_out.view(x.shape[0], -1)
        kernel_out = self.fc_kernel(kernel_feat)

        # Combine
        combined = torch.cat([cnn_out, kernel_out], dim=1)
        logits = self.fc_combined(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumHybridNAT"]
