import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerAutoEncoder(nn.Module):
    """
    Classical hybrid sampler that combines a lightweight autoencoder with a
    parametric sampler network. The encoder reduces the 2‑dimensional input into
    a latent space and the decoder reconstructs a 2‑dimensional probability
    distribution via softmax. The network is designed to be compatible with
    the quantum helper defined in the corresponding QML module and can be
    used as a baseline for comparison.
    """
    def __init__(self, latent_dim: int = 4, hidden_dims: tuple[int, int] = (64, 32)):
        super().__init__()
        encoder = []
        in_dim = 2
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            in_dim = h
        decoder.append(nn.Linear(in_dim, 2))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        out = self.decoder(latent)
        return F.softmax(out, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
