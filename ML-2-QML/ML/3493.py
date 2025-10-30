import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class _CNNFeatureExtractor(nn.Module):
    """Classic 2‑layer CNN followed by a small fully‑connected encoder."""
    def __init__(self, in_channels: int = 1, out_features: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.conv(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.bn(out)

class _QuantumLinearHead(nn.Module):
    """Simulated quantum linear head using classical layers."""
    def __init__(self, in_features: int, out_features: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.readout = nn.Linear(32, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        out = self.readout(out)
        return self.bn(out)

class QuantumHybridNat(nn.Module):
    """
    Hybrid model that supports both image classification and sequence tagging.
    - mode='image' : uses a CNN feature extractor followed by a quantum‑style head.
    - mode='sequence' : uses an embedding layer and a classical LSTM.
    """
    def __init__(
        self,
        mode: str = "image",
        *,
        in_channels: int = 1,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        hidden_dim: int = 128,
        tagset_size: Optional[int] = None,
        use_quantum_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "image":
            self.extractor = _CNNFeatureExtractor(in_channels=in_channels, out_features=64)
            self.head = _QuantumLinearHead(in_features=64, out_features=4)
        elif mode == "sequence":
            assert vocab_size is not None and embedding_dim is not None and tagset_size is not None
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            # Classical LSTM serves as fallback; the flag is ignored in this pure‑classical module
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "image":
            feat = self.extractor(x)
            return self.head(feat)
        elif self.mode == "sequence":
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.classifier(lstm_out)
            return F.log_softmax(logits, dim=-1)
        else:
            raise RuntimeError("Forward called with unknown mode")

__all__ = ["QuantumHybridNat"]
