import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

class ClassicalSelfAttention(nn.Module):
    """Classical self‑attention block mirroring the original seed but implemented in PyTorch."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: torch.Tensor):
        R = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        E = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()
        query = inputs @ R
        key   = inputs @ E
        value = inputs
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Compact MLP auto‑encoder used as a compression backbone."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers += [nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Dropout(cfg.dropout) if cfg.dropout>0 else nn.Identity()]
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers += [nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Dropout(cfg.dropout) if cfg.dropout>0 else nn.Identity()]
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class SelfAttentionHybrid(nn.Module):
    """
    A unified classical pipeline that chains self‑attention, an auto‑encoder, and a regression head.
    The architecture is inspired by the four reference pairs:
    * Self‑attention mechanics (pair 1)
    * Auto‑encoder compression (pair 2)
    * Regression mapping (pair 3)
    * NAT‑style fully‑connected projection (pair 4)
    """
    def __init__(self,
                 embed_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.attn = ClassicalSelfAttention(embed_dim)
        self.autoenc = AutoencoderNet(AutoencoderConfig(embed_dim, latent_dim, hidden_dims, dropout))
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation matrix parameters for the attention query/key.
        entangle_params : np.ndarray
            Flattened entanglement matrix parameters for the attention key.
        inputs : torch.Tensor
            Batch of input feature vectors (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Regressed scalar per example (batch,).
        """
        attn_out = self.attn(rotation_params, entangle_params, inputs)
        latent = self.autoenc.encode(attn_out)
        out = self.regressor(latent)
        return out.squeeze(-1)

    def train_on(self,
                 data_loader,
                 lr: float = 1e-3,
                 epochs: int = 50,
                 device: torch.device | None = None) -> list[float]:
        """
        A minimal training loop that optimises the attention, auto‑encoder and regression head jointly.
        The loss is MSE against a dummy target supplied by the data loader.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in data_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = self.forward(batch['rotation'], batch['entangle'], inputs)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(data_loader.dataset)
            history.append(epoch_loss)
        return history
