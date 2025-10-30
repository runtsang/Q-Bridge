from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

# --------------------------------------------------------------------------- #
# 1.  Auto‑encoder components
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# 2.  Classical LSTM (drop‑in replacement)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Gate linear layers
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# 3.  Hybrid estimator
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """
    Hybrid estimator combining an auto‑encoder, a (classical) LSTM tagger,
    and a regression head.  The architecture can be toggled to use a
    purely classical pipeline or a quantum‑enhanced LSTM by passing
    ``n_qubits > 0``.  The auto‑encoder projects the raw inputs into a
    latent space that serves as the input to the LSTM.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 ae_latent_dim: int | None = None,
                 ae_dropout: float = 0.0) -> None:
        super().__init__()
        # Optional auto‑encoder
        self.autoencoder = None
        if ae_latent_dim is not None:
            self.autoencoder = Autoencoder(input_dim,
                                           latent_dim=ae_latent_dim,
                                           hidden_dims=(128, 64),
                                           dropout=ae_dropout)

        # LSTM tagger (classical or quantum)
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim)

        # Final regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Auto‑encoder projection
        if self.autoencoder is not None:
            data = self.autoencoder.encode(data)
        # LSTM processing
        lstm_out, _ = self.lstm(data.unsqueeze(0))
        # Regression
        return self.head(lstm_out.squeeze(0))

__all__ = ["EstimatorQNN", "Autoencoder", "AutoencoderNet", "AutoencoderConfig", "QLSTM"]
