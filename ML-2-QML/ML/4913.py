"""Hybrid classical sampler network combining convolution, autoencoder, and QCNN layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Convolutional filter emulating a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce spatial dimensions to a single scalar per sample
        return activations.mean(dim=[2, 3])

class AutoencoderNet(nn.Module):
    """Autoencoder with configurable latent dimension."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QCNNBlock(nn.Module):
    """QCNN‑style fully‑connected block."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class SamplerQNNGen(nn.Module):
    """Hybrid sampler network that mirrors the quantum architecture."""
    def __init__(self, input_shape: tuple[int, int, int] = (1, 2, 2), latent_dim: int = 32):
        super().__init__()
        # Convolution filter
        self.conv = ConvFilter(kernel_size=2, threshold=0.0)

        # After conv we obtain a scalar per sample
        flat_dim = 1

        # Autoencoder bottleneck
        self.autoencoder = AutoencoderNet(
            input_dim=flat_dim,
            latent_dim=latent_dim,
        )

        # QCNN‑style stack
        self.feature_map = QCNNBlock(2, 4)
        self.conv1 = QCNNBlock(4, 4)
        self.pool1 = QCNNBlock(4, 3)
        self.conv2 = QCNNBlock(3, 2)
        self.pool2 = QCNNBlock(2, 1)
        self.conv3 = QCNNBlock(1, 1)
        self.head = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, 1, 2, 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        # Convolution + reduction
        conv_out = self.conv(x).unsqueeze(-1).unsqueeze(-1)  # shape: (B,1,1,1)
        flat = conv_out.view(conv_out.size(0), -1)  # (B,1)

        # Encode to latent space
        latent = self.autoencoder.encode(flat)

        # QCNN hierarchy
        y = self.feature_map(latent)
        y = self.conv1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.conv3(y)

        logits = self.head(y)
        return F.softmax(logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation of the input."""
        conv_out = self.conv(x).unsqueeze(-1).unsqueeze(-1)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.autoencoder.encode(flat)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent representation."""
        return self.autoencoder.decode(latent)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction pipeline."""
        latent = self.encode(x)
        return self.decode(latent)

__all__ = ["SamplerQNNGen"]
