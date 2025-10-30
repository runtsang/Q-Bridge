import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

# Import the quantum decoder from the separate QML module
from autoencoder_qm import AutoencoderQEMerge as QuantumDecoder

class AutoencoderQEMergeConfig:
    """Configuration for the hybrid autoencoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple = (128, 64),
                 dropout: float = 0.1,
                 attention_heads: int = 4,
                 attention_dim: int = 4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim

class ClassicalEncoder(nn.Module):
    """Encoder MLP that maps input to latent space."""
    def __init__(self, config: AutoencoderQEMergeConfig):
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SelfAttentionLayer(nn.Module):
    """Simple self‑attention block operating on the latent code."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class AutoencoderQEMerge(nn.Module):
    """Hybrid autoencoder: classical encoder + self‑attention + quantum decoder."""
    def __init__(self, config: AutoencoderQEMergeConfig):
        super().__init__()
        self.encoder  = ClassicalEncoder(config)
        self.attention = SelfAttentionLayer(config.attention_dim)
        # Quantum decoder is imported from the QML module
        self.decoder  = QuantumDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        attn_out = self.attention(latent)
        # The quantum decoder expects a NumPy array
        recon_np = self.decoder.decode(attn_out.detach().cpu().numpy())
        return torch.tensor(recon_np, device=x.device, dtype=x.dtype)

def train_autoencoder_qe(model: AutoencoderQEMerge,
                         data: torch.Tensor,
                         *,
                         epochs: int = 50,
                         batch_size: int = 64,
                         lr: float = 1e-3) -> list[float]:
    """Train the hybrid autoencoder and return the loss history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
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
    """Ensure we are working with a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["AutoencoderQEMergeConfig",
           "AutoencoderQEMerge",
           "train_autoencoder_qe"]
