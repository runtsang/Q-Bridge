import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Callable, Optional

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class QuanvolutionFilter(nn.Module):
    """A classical 2×2 convolution filter that mimics the structure of a quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        # Use a slightly larger kernel to capture more context.
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class AutoencoderNet(nn.Module):
    """Hybrid autoencoder that can optionally use a quantum encoder."""
    def __init__(self,
                 config: AutoencoderConfig,
                 quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__()
        self.quantum_encoder = quantum_encoder

        # Classical encoder
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

        # Classical decoder
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs either via the quantum encoder or the classical encoder."""
        if self.quantum_encoder is not None:
            return self.quantum_encoder(inputs)
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int,
                *,
                latent_dim: int = 32,
                hidden_dims: Tuple[int,...] = (128, 64),
                dropout: float = 0.1,
                quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> AutoencoderNet:
    """Factory for a hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config, quantum_encoder=quantum_encoder)

def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> list[float]:
    """Train the autoencoder, returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
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
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder", "QuanvolutionFilter"]
