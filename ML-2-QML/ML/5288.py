import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    """
    Classical auto‑encoder with optional quantum latent mapping and
    optional sequence processing.  The architecture mirrors the
    quantum‑enhanced references but is fully implementable with
    PyTorch.
    """
    def __init__(
        self,
        config: AutoencoderConfig,
        use_quantum: bool = False,
        use_sequence: bool = False,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_quantum = use_quantum
        self.use_sequence = use_sequence

        # Encoder
        self.encoder = self._build_encoder()

        # Latent mapping – classical linear for both regimes
        self.latent_layer = nn.Linear(config.latent_dim, config.latent_dim)

        # Sequence module
        if self.use_sequence:
            self.lstm = nn.LSTM(
                input_size=config.latent_dim,
                hidden_size=config.latent_dim,
                batch_first=True,
            )
        else:
            self.lstm = None

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.config.input_dim
        for hidden in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.config.latent_dim
        for hidden in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  For static data x has shape (batch, input_dim).
        For sequences x has shape (batch, seq_len, input_dim) when
        `use_sequence=True`.
        """
        if self.use_sequence:
            batch, seq_len, _ = x.shape
            flat = x.reshape(batch * seq_len, -1)
            encoded = self.encoder(flat)
            latent = self.latent_layer(encoded)
            latent = latent.reshape(batch, seq_len, -1)
            lstm_out, _ = self.lstm(latent)
            decoded = self.decoder(lstm_out)
            return decoded.reshape(batch, seq_len, -1)
        else:
            encoded = self.encoder(x)
            latent = self.latent_layer(encoded)
            decoded = self.decoder(latent)
            return decoded
