import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid transformer‑autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    transformer_heads: int = 4
    transformer_ffn: int = 256
    transformer_blocks: int = 2

class TransformerEncoder(nn.Module):
    """Transformer encoder block."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn,
            dropout=config.dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_blocks)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(src)

class TransformerDecoder(nn.Module):
    """Transformer decoder block."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.input_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn,
            dropout=config.dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.transformer_blocks)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, memory)

class AutoencoderGen475(nn.Module):
    """
    Hybrid transformer‑based autoencoder that emits a low‑dimensional latent vector
    which can be fed into a quantum autoencoder circuit.
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.latent_map = nn.Linear(config.input_dim, config.latent_dim)
        self.expand_map = nn.Linear(config.latent_dim, config.input_dim)
        self.decoder = TransformerDecoder(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space."""
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.latent_map(pooled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to input space."""
        expanded = self.expand_map(latent)
        seq_len = self.config.input_dim
        tgt = expanded.unsqueeze(1).repeat(1, seq_len, 1)
        memory = self.encoder(torch.zeros_like(tgt))
        return self.decoder(tgt, memory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the low‑dimensional latent vector."""
        return self.encode(x)

def _as_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_autoencoder(
    model: AutoencoderGen475,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
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
