"""
Hybrid autoencoder that fuses a classical CNN encoder with a quantum‑inspired fully‑connected layer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Classical QFCModel (from QuantumNAT) – used as the feature extractor
# --------------------------------------------------------------------------- #
class QFCModel(nn.Module):
    """Simple CNN followed by a fully connected projection to four features."""
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
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# Hybrid Autoencoder definition
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """
    Encoder: QFCModel → linear → latent space.
    Decoder: linear → reshape → deconvolution to reconstruct image.
    """
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.qfc = QFCModel()

        # Linear bridge between 4‑feature output and latent space
        self.enc_linear = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.ReLU()
        )

        # Decoder mirrors the encoder
        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU()
        )

        # Reconstruct to 28x28 image
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfc(x)          # shape: (N, 4)
        latent = self.enc_linear(features)  # shape: (N, latent_dim)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.dec_linear(z)        # shape: (N, 4)
        out = out.view(-1, 4, 1, 1)     # reshape for deconv
        return self.deconv(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Factory and training utilities
# --------------------------------------------------------------------------- #
def HybridAutoencoder(latent_dim: int = 32) -> HybridAutoencoderNet:
    """Convenience factory mirroring the original Autoencoder helper."""
    return HybridAutoencoderNet(latent_dim=latent_dim)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Helper: convert data to float32 tensor
# --------------------------------------------------------------------------- #
def _as_tensor(data: torch.Tensor | torch.Tensor | list | tuple | None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "train_hybrid_autoencoder"]
