import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data):
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(data, dtype=torch.float32)
    if t.dtype!= torch.float32:
        t = t.to(dtype=torch.float32)
    return t

class SelfAttentionLayer(nn.Module):
    """Classical self‑attention block compatible with the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

class AutoencoderHybridNet(nn.Module):
    """Dense autoencoder enhanced with a self‑attention layer."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        attention_dim: int = 64,
    ):
        super().__init__()
        # Encoder
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

        # Attention
        self.attention = SelfAttentionLayer(attention_dim)

        # Decoder
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
        z = self.attention(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder(
    net: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that records the reconstruction loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = net(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

class AutoencoderHybrid:
    """Facade exposing the dense autoencoder with attention."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        attention_dim: int = 64,
    ):
        self.net = AutoencoderHybridNet(input_dim, latent_dim, hidden_dims, dropout, attention_dim)

    def encode(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.net.encode(_as_tensor(x))

    def decode(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.net.decode(_as_tensor(z))

    def fit(
        self,
        data: np.ndarray | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> list[float]:
        return train_autoencoder(self.net, data, epochs=epochs, batch_size=batch_size, lr=lr)

    def reconstruct(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self.net.forward(_as_tensor(x))

__all__ = ["AutoencoderHybrid", "train_autoencoder"]
