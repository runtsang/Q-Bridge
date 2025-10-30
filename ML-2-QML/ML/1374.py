import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class QuantumNATEnhanced(nn.Module):
    """Hybrid classical‑quantum encoder with multi‑task head.

    The architecture expands on the original QFCModel by
    1. Adding a learnable 2‑layer CNN encoder that **pre‑processes** the
       input image before quantum‑phase‑based feature extraction.
    2. Replacing the fully‑connected projection to 4‑features
       with **two heads**: one for classification (4 classes) and
       a 2‑dimensional reconstruction vector.
    The design is fully compatible with PyTorch‑Torchvision datasets.
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten size after two 2‑poolings on 28×28 input
        self.flatten_size = 32 * 7 * 7

        # Quantum feature extractor (classical proxy)
        self.q_module = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        # Batch‑norm on the quantum‑like features
        self.batch_norm = BatchNorm1d(4)

        # Multi‑task heads
        self.class_head = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 4)
        )
        self.recon_head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        enc = self.encoder(x)
        flat = enc.view(bsz, -1)
        q_feat = self.q_module(flat)
        q_feat = self.batch_norm(q_feat)
        cls = self.class_head(q_feat)
        recon = self.recon_head(q_feat)
        return cls, recon
