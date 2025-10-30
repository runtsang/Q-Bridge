import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# ------------------------------------------------------------------
# Classical building blocks inspired by the reference seeds
# ------------------------------------------------------------------


class FCLayer(nn.Module):
    """Fully connected layer that mimics the quantum FCL example."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # ``thetas`` are treated as weights for a single linear unit
        return torch.tanh(self.linear(thetas)).mean(dim=0, keepdim=True)


class QCNNModel(nn.Module):
    """Stack of linear layers emulating a quantum convolution‑like network."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class QuanvolutionFilter(nn.Module):
    """Classical convolution filter inspired by the Quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


# ------------------------------------------------------------------
# Hybrid autoencoder that stitches the components together
# ------------------------------------------------------------------


@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quanvolution: bool = False
    use_qcnn: bool = True


class HybridAutoencoder(nn.Module):
    """A hybrid autoencoder that blends classical, QCNN, FCL and Quanvolution layers."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Optional 2‑D feature extraction
        self.quanvolution = QuanvolutionFilter() if cfg.use_quanvolution else None

        # Encoder pipeline
        encoder_layers = []
        in_dim = cfg.input_dim
        if cfg.use_qcnn:
            # Replace the first dense block with a QCNN‑style feature extractor
            self.qcnn = QCNNModel()
            encoder_layers.append(self.qcnn)
            in_dim = 1  # QCNN outputs a single scalar
        else:
            self.qcnn = None

        # Standard dense encoder
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Optional quantum fully‑connected layer (mocked classically)
        self.fcl = FCLayer(n_features=cfg.latent_dim)

        # Decoder pipeline
        decoder_layers = []
        in_dim = cfg.latent_dim
        decoder_layers.append(nn.Linear(in_dim, cfg.hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        if cfg.dropout > 0.0:
            decoder_layers.append(nn.Dropout(cfg.dropout))
        for hidden in reversed(cfg.hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quanvolution is not None:
            # Expect 4‑D tensor: (B, C, H, W)
            inputs = self.quanvolution(inputs)
        if self.qcnn is not None:
            inputs = self.qcnn(inputs)
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Pass through optional quantum fully‑connected layer
        if self.fcl is not None:
            latents = self.fcl(latents)
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quanvolution: bool = False,
    use_qcnn: bool = True,
) -> HybridAutoencoder:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quanvolution=use_quanvolution,
        use_qcnn=use_qcnn,
    )
    return HybridAutoencoder(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
