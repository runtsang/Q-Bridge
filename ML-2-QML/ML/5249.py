from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class AutoencoderConfig:
    """Configuration for the auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight MLP auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

class SamplerModule(nn.Module):
    """Softmax sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

class EstimatorNN(nn.Module):
    """Regressor network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

class FraudDetectionHybrid(nn.Module):
    """Hybrid classical fraud detection pipeline."""
    def __init__(self, autoenc_cfg: AutoencoderConfig):
        super().__init__()
        self.autoencoder = AutoencoderNet(autoenc_cfg)
        self.sampler = SamplerModule()
        self.estimator = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, sample and estimate fraud probability."""
        z = self.autoencoder.encode(x)
        # use the first two latent dimensions as features for the sampler
        probs = self.sampler(z[:, :2])
        score = self.estimator(probs)
        return score

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> list[float]:
        """Train only the auto‑encoder."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(data.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "SamplerModule",
    "EstimatorNN",
    "FraudDetectionHybrid",
]
