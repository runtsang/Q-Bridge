import torch
from torch import nn
from typing import Callable, Tuple

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder integrating a classical encoder with a plug‑in quantum decoder.

    Parameters
    ----------
    encoder : nn.Module
        A PyTorch module that maps input features to a latent representation.
    decoder : Callable[[torch.Tensor], torch.Tensor] | None
        Function that receives a latent tensor and returns a reconstruction.
        If ``None`` a trivial identity decoder is used.
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder or (lambda x: x)

    def set_decoder(self, decoder: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Replace the quantum decoder after construction."""
        self.decoder = decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def build_classical_encoder(input_dim: int,
                            latent_dim: int = 32,
                            hidden_dims: Tuple[int,...] = (128, 64),
                            dropout: float = 0.1) -> nn.Module:
    """Constructs a feed‑forward encoder mirroring the original Autoencoder."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = h
    layers.append(nn.Linear(in_dim, latent_dim))
    return nn.Sequential(*layers)

class EstimatorNN(nn.Module):
    """Small feed‑forward regressor used as a post‑processing head."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

def estimator_qnn_decoder() -> nn.Module:
    """Return an instance of the EstimatorNN to be used as a classical decoder."""
    return EstimatorNN()

__all__ = ["HybridAutoencoder", "build_classical_encoder", "estimator_qnn_decoder", "EstimatorNN"]
