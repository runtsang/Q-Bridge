import torch
import numpy as np
from torch import nn
from typing import Any, Dict

class UnifiedQuantumClassicLayer(nn.Module):
    """
    Unified classical module that can act as a fully‑connected layer (FC),
    self‑attention (SA), convolution‑like stack (CNN), or auto‑encoder (AE)
    depending on the ``mode`` argument.
    """

    def __init__(self, mode: str = "FC", **kwargs: Any):
        super().__init__()
        self.mode = mode.upper()
        self.config: Dict[str, Any] = kwargs

        if self.mode == "FC":
            n_features = kwargs.get("n_features", 1)
            self.model = nn.Linear(n_features, 1)

        elif self.mode == "SA":
            embed_dim = kwargs.get("embed_dim", 4)
            self.query_proj = nn.Linear(embed_dim, embed_dim)
            self.key_proj   = nn.Linear(embed_dim, embed_dim)
            self.value_proj = nn.Linear(embed_dim, embed_dim)

        elif self.mode == "CNN":
            # A lightweight convolution‑style stack inspired by QCNN.py
            self.model = nn.Sequential(
                nn.Linear(8, 16), nn.Tanh(),
                nn.Linear(16, 16), nn.Tanh(),
                nn.Linear(16, 12), nn.Tanh(),
                nn.Linear(12, 8), nn.Tanh(),
                nn.Linear(8, 4), nn.Tanh(),
                nn.Linear(4, 4), nn.Tanh(),
                nn.Linear(4, 1)
            )

        elif self.mode == "AE":
            input_dim   = kwargs.get("input_dim", 10)
            latent_dim  = kwargs.get("latent_dim", 32)
            hidden_dims = kwargs.get("hidden_dims", (128, 64))
            self.encoder = nn.Sequential()
            in_dim = input_dim
            for hidden in hidden_dims:
                self.encoder.add_module(f"enc_lin_{hidden}", nn.Linear(in_dim, hidden))
                self.encoder.add_module(f"enc_relu_{hidden}", nn.ReLU())
                in_dim = hidden
            self.encoder.add_module("enc_latent", nn.Linear(in_dim, latent_dim))

            self.decoder = nn.Sequential()
            in_dim = latent_dim
            for hidden in reversed(hidden_dims):
                self.decoder.add_module(f"dec_lin_{hidden}", nn.Linear(in_dim, hidden))
                self.decoder.add_module(f"dec_relu_{hidden}", nn.ReLU())
                in_dim = hidden
            self.decoder.add_module("dec_out", nn.Linear(in_dim, input_dim))

        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "FC":
            return self.model(x)

        elif self.mode == "SA":
            q = self.query_proj(x)
            k = self.key_proj(x)
            v = self.value_proj(x)
            scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
            return scores @ v

        elif self.mode == "CNN":
            return self.model(x)

        elif self.mode == "AE":
            lat = self.encoder(x)
            return self.decoder(lat)

        else:
            raise RuntimeError("Invalid mode during forward pass")

__all__ = ["UnifiedQuantumClassicLayer"]
