import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

class Kernel(nn.Module):
    """Classical RBF kernel for regularization."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

@dataclass
class AutoencoderGen044Config:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_gamma: float = 1.0

class AutoencoderGen044(nn.Module):
    """Hybrid autoencoder that adds an RBF kernel regularizer over the latent space."""
    def __init__(self, config: AutoencoderGen044Config, reference_vectors: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        # Kernel module
        self.kernel = Kernel(config.kernel_gamma)
        if reference_vectors is not None:
            self.register_buffer('reference_vectors', reference_vectors.clone())
        else:
            self.register_buffer('reference_vectors', None)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    def compute_kernel_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel matrix between x and the stored reference vectors."""
        if self.reference_vectors is None:
            raise ValueError("Reference vectors not set")
        return self.kernel(x.unsqueeze(1), self.reference_vectors.unsqueeze(0)).squeeze(-1)

    def set_reference_vectors(self, refs: torch.Tensor) -> None:
        """Store reference latent vectors for kernel computation."""
        self.register_buffer('reference_vectors', refs.clone())

    def kernel_regularization(self, latent: torch.Tensor) -> torch.Tensor:
        """Return a scalar regularization term based on kernel similarity to references."""
        k_mat = self.compute_kernel_matrix(latent)
        return torch.mean(k_mat)

__all__ = ["AutoencoderGen044", "AutoencoderGen044Config", "Kernel"]
