import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Any

def _as_tensor(data: Iterable[Any] | torch.Tensor) -> torch.Tensor:
    """Convert data to a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class HybridAutoEncoder(nn.Module):
    """Hybrid auto‑encoder that merges a convolutional feature extractor, a fully‑connected encoder,
    a quantum‑style latent transform, and a symmetrical decoder."""
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape

        # Convolutional encoder (Quantum‑NAT style)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        conv_out_dim = 16 * (input_shape[1] // 4) * (input_shape[2] // 4)

        # Fully‑connected encoder
        encoder_layers = []
        in_dim = conv_out_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder_fc = nn.Sequential(*encoder_layers)

        # Quantum‑style latent layer (simple linear mapping)
        self.quantum_latent = nn.Linear(latent_dim, latent_dim)

        # Fully‑connected decoder
        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, conv_out_dim))
        self.decoder_fc = nn.Sequential(*decoder_layers)

        # Transposed conv decoder to reconstruct image
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, input_shape[0], kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv_encoder(x)
        flat = conv.view(conv.size(0), -1)
        latent = self.encoder_fc(flat)
        return self.quantum_latent(latent)

    def decode(self, qlatent: torch.Tensor) -> torch.Tensor:
        flat = self.decoder_fc(qlatent)
        feature_map = flat.view(-1, 16, self.input_shape[1] // 4, self.input_shape[2] // 4)
        return self.deconv(feature_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Standard reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = ["HybridAutoEncoder", "train_hybrid_autoencoder"]
