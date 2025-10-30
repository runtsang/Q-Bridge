import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class ConvFilter(nn.Module):
    """Classical convolutional filter mimicking a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, 1, H, W) or (1, H, W) for a single sample.

        Returns
        -------
        torch.Tensor
            Activated and optionally dropped‑out feature map.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return self.dropout(activations)

    def run(self, data: torch.Tensor | list[list[float]]) -> float:
        """Convenience wrapper for a single sample."""
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(tensor).mean().item()


class AutoencoderConfig:
    """Configuration holder for the autoencoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable depth."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*self._build_layers(config.input_dim,
                                                          config.hidden_dims,
                                                          config.latent_dim,
                                                          config.dropout,
                                                          encoder=True))
        self.decoder = nn.Sequential(*self._build_layers(config.latent_dim,
                                                          config.hidden_dims[::-1],
                                                          config.input_dim,
                                                          config.dropout,
                                                          encoder=False))

    def _build_layers(self,
                      in_dim: int,
                      hidden_dims: tuple[int,...],
                      out_dim: int,
                      dropout: float,
                      encoder: bool) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class HybridConvAutoencoder:
    """Hybrid encoder–decoder that chains a conv filter with an autoencoder."""
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 conv_dropout: float = 0.0,
                 ae_config: AutoencoderConfig | None = None) -> None:
        self.conv = ConvFilter(kernel_size=conv_kernel,
                               threshold=conv_threshold,
                               dropout=conv_dropout)
        if ae_config is None:
            raise ValueError("Autoencoder configuration must be provided.")
        self.autoencoder = AutoencoderNet(ae_config)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the conv filter, flatten, and pass through the autoencoder encoder.
        """
        conv_out = self.conv(data)          # shape (batch, 1, k, k)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.autoencoder.encode(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(data))

    def train_autoencoder(self,
                          data: torch.Tensor,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          device: torch.device | None = None) -> list[float]:
        """
        Train only the autoencoder part, keeping the conv filter fixed.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
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


__all__ = ["ConvFilter", "AutoencoderConfig", "AutoencoderNet", "HybridConvAutoencoder"]
