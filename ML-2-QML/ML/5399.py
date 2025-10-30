import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerModule(nn.Module):
    """Feed‑forward sampler that maps 2‑dimensional inputs to a 2‑dimensional probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QFCModel(nn.Module):
    """CNN followed by a fully‑connected head, inspired by Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

class QuanvolutionFilter(nn.Module):
    """2×2 patch‑wise convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        return f.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier built on top of the quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.filter(x)
        logits = self.linear(f)
        return F.log_softmax(logits, dim=-1)

def ClassicalSelfAttention(embed_dim: int = 4):
    """Simple self‑attention that operates on NumPy arrays."""
    class _SelfAttention:
        def __init__(self) -> None:
            self.embed_dim = embed_dim
        def run(self, rot: np.ndarray, ent: np.ndarray, inputs: np.ndarray) -> np.ndarray:
            query = torch.tensor(inputs @ rot.reshape(self.embed_dim, -1), dtype=torch.float32)
            key   = torch.tensor(inputs @ ent.reshape(self.embed_dim, -1), dtype=torch.float32)
            value = torch.tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()
    return _SelfAttention()

class HybridSamplerQNN(nn.Module):
    """Classical hybrid encoder that chains sampler, CNN, quanvolution and attention."""
    def __init__(self) -> None:
        super().__init__()
        self.sampler = SamplerModule()
        self.qfc     = QFCModel()
        self.filter  = QuanvolutionFilter()
        self.attn    = ClassicalSelfAttention()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be (batch, 2) for the sampler
        samp = self.sampler(x)
        # create a dummy 28×28 gray‑scale image from the two features
        img = torch.zeros(x.size(0), 1, 28, 28, device=x.device)
        img[:, 0, 0, 0] = x[:, 0]
        img[:, 0, 0, 1] = x[:, 1]
        qfc_out = self.qfc(img)
        quanv_out = self.filter(img)
        # concatenate all feature maps
        feat = torch.cat([samp, qfc_out, quanv_out], dim=1)
        # apply classical self‑attention
        rot = np.random.randn(12)   # 4 × 3 rotation parameters
        ent = np.random.randn(3)    # 4‑1 entanglement parameters
        att = self.attn.run(rot, ent, feat.detach().cpu().numpy())
        return torch.from_numpy(att).to(x.device).float()
