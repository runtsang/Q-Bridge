from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence, Tuple

class HybridLayer(nn.Module):
    """Hybrid classical neural module supporting FCL, GraphQNN, Autoencoder, and LSTM modes."""
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == "fcl":
            n_features = kwargs.get("n_features", 1)
            self.linear = nn.Linear(n_features, 1)
        elif mode == "graph":
            arch = kwargs.get("arch", [2, 2, 1])
            self.arch = arch
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.randn(out, in_)) for in_, out in zip(arch[:-1], arch[1:])]
            )
        elif mode == "autoencoder":
            input_dim = kwargs.get("input_dim", 1)
            latent_dim = kwargs.get("latent_dim", 32)
            hidden_dims = kwargs.get("hidden_dims", (128, 64))
            dropout = kwargs.get("dropout", 0.1)

            encoder_layers = []
            in_dim = input_dim
            for hidden in hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if dropout > 0.0:
                    encoder_layers.append(nn.Dropout(dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            in_dim = latent_dim
            for hidden in reversed(hidden_dims):
                decoder_layers.append(nn.Linear(in_dim, hidden))
                decoder_layers.append(nn.ReLU())
                if dropout > 0.0:
                    decoder_layers.append(nn.Dropout(dropout))
                in_dim = hidden
            decoder_layers.append(nn.Linear(in_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        elif mode == "lstm":
            input_dim = kwargs.get("input_dim", 1)
            hidden_dim = kwargs.get("hidden_dim", 1)
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fcl":
            return torch.tanh(self.linear(x)).mean(dim=0)
        elif self.mode == "graph":
            out = x
            for w in self.weights:
                out = torch.tanh(w @ out)
            return out
        elif self.mode == "autoencoder":
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        elif self.mode == "lstm":
            out, _ = self.lstm(x.unsqueeze(0))
            return out.squeeze(0)
        else:
            raise RuntimeError("Invalid mode")

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        return self.forward(torch.tensor(thetas, dtype=torch.float32))

def FCL(n_features: int = 1) -> HybridLayer:
    """Return a fully‑connected layer instance."""
    return HybridLayer(mode="fcl", n_features=n_features)

def GraphQNN(arch: Sequence[int] = (2, 2, 1), samples: int = 10) -> HybridLayer:
    """Return a graph‑based neural network instance."""
    return HybridLayer(mode="graph", arch=arch)

def Autoencoder(
    input_dim: int = 1,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridLayer:
    """Return an auto‑encoder instance."""
    return HybridLayer(
        mode="autoencoder",
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

def QLSTM(input_dim: int = 1, hidden_dim: int = 1) -> HybridLayer:
    """Return a classical LSTM instance."""
    return HybridLayer(mode="lstm", input_dim=input_dim, hidden_dim=hidden_dim)

__all__ = ["HybridLayer", "FCL", "GraphQNN", "Autoencoder", "QLSTM"]
