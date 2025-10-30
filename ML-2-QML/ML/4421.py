import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Classical building blocks
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Drop‑in replacement for a quanvolutional filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with configurable depth."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class QFCModel(nn.Module):
    """CNN followed by a fully‑connected projection, inspired by Quantum‑NAT."""
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
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


class SamplerModule(nn.Module):
    """Classical sampler that outputs a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# ------------------------------------------------------------------
# Hybrid Sampler
# ------------------------------------------------------------------
class SamplerQNNGen099:
    """Hybrid sampler that chains convolution, auto‑encoding, a QFC projection,
    and a classical sampler.  All stages are optional and can be toggled
    at construction time."""
    def __init__(self,
                 use_conv: bool = True,
                 use_autoencoder: bool = True,
                 use_qfc: bool = True,
                 use_sampler: bool = True,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 ae_input_dim: int = 784,
                 ae_latent_dim: int = 32,
                 ae_hidden: tuple[int, int] = (128, 64),
                 ae_dropout: float = 0.1) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.use_autoencoder = use_autoencoder
        self.use_qfc = use_qfc
        self.use_sampler = use_sampler

        if self.use_conv:
            self.conv = ConvFilter(kernel_size=conv_kernel,
                                   threshold=conv_threshold)
        if self.use_autoencoder:
            self.autoencoder = AutoencoderNet(
                input_dim=ae_input_dim,
                latent_dim=ae_latent_dim,
                hidden_dims=ae_hidden,
                dropout=ae_dropout
            )
        if self.use_qfc:
            self.qfc = QFCModel()
        if self.use_sampler:
            self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the enabled stages and return a 2‑class
        probability distribution."""
        out = x
        if self.use_conv:
            out = self.conv(out)
        if self.use_autoencoder:
            out = self.autoencoder.encode(out)
        if self.use_qfc:
            out = self.qfc(out)
        if self.use_sampler:
            out = self.sampler(out)
        return out


def SamplerQNNGen099_factory(**kwargs) -> SamplerQNNGen099:
    """Convenience factory that mirrors the original SamplerQNN() API."""
    return SamplerQNNGen099(**kwargs)


__all__ = ["SamplerQNNGen099", "SamplerQNNGen099_factory"]
