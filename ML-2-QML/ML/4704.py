import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Classical convolutional encoder
# --------------------------------------------------------------------------- #
class ConvEncoder(nn.Module):
    """Feature extractor that mirrors the structure of QFCModel’s ConvNet
    but is kept deliberately shallow so the quantum core can operate on
    a small latent space."""
    def __init__(self, out_features: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.project = nn.Linear(16 * 7 * 7, out_features)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        flat = feats.view(x.shape[0], -1)
        return self.norm(self.project(flat))

# --------------------------------------------------------------------------- #
#  Classical auto‑encoder (ML‑only)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class ClassicalAutoencoder(nn.Module):
    """Simple MLP auto‑encoder that maps `input_dim` → `latent_dim` → `input_dim`."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
#  Classical LSTM (drop‑in replacement for the quantum version)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Linear‑gated LSTM that mimics the interface of the quantum QLSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
#  Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedQuantumAutoLSTM(nn.Module):
    """Combines a CNN encoder, a classical auto‑encoder, and a classical LSTM."""
    def __init__(
        self,
        conv_out_features: int = 64,
        ae_latent_dim: int = 32,
        lstm_hidden_dim: int = 64,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(conv_out_features)
        self.autoencoder = ClassicalAutoencoder(
            AutoencoderConfig(input_dim=conv_out_features, latent_dim=ae_latent_dim)
        )
        self.lstm = ClassicalQLSTM(ae_latent_dim, lstm_hidden_dim)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, 1, 28, 28) representing a batch of image sequences.

        Returns
        -------
        torch.Tensor
            Log‑softmax logits over `num_classes`.
        """
        B, seq_len, C, H, W = x.shape
        # Encode each image in the sequence
        feats = self.encoder(x.view(B * seq_len, C, H, W))  # (B*seq_len, feat_dim)
        # Compress with the auto‑encoder
        latents = self.autoencoder.encode(feats)  # (B*seq_len, latent_dim)
        latents = latents.view(B, seq_len, -1)
        # LSTM expects (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(latents.permute(1, 0, 2))
        # Use the last hidden state
        hidden = lstm_out[-1]
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=1)

__all__ = ["UnifiedQuantumAutoLSTM"]
