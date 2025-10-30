import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    num_encoder_blocks: int = 2
    num_decoder_blocks: int = 2
    dropout: float = 0.1
    use_lstm: bool = False
    hidden_dim: int = 64

class MultiHeadAttention(nn.Module):
    """Classical multi‑head attention used inside TransformerBlock."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForward(nn.Module):
    """Simple two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Self‑attention block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridAutoencoder(nn.Module):
    """Classical transformer‑based autoencoder with optional LSTM support."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            *[TransformerBlock(config.embed_dim, config.num_heads, config.ffn_dim, config.dropout)
              for _ in range(config.num_encoder_blocks)]
        )

        # Latent projection
        self.latent = nn.Linear(config.embed_dim, config.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            *[TransformerBlock(config.latent_dim, config.num_heads, config.ffn_dim, config.dropout)
              for _ in range(config.num_decoder_blocks)]
        )
        self.output = nn.Linear(config.latent_dim, config.input_dim)

        # Optional LSTM for sequence data
        if config.use_lstm:
            self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, batch_first=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return self.output(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_autoencoder(
    model: HybridAutoencoder,
    data: np.ndarray | torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Reconstruction training loop returning loss history."""
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

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)
