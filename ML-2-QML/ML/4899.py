"""Hybrid fraud detection module combining classical autoencoder, self‑attention, and quantum‑inspired fully‑connected layer.

The class :class:`FraudDetectionHybrid` inherits from :class:`torch.nn.Module` and exposes a
``forward`` method that accepts a tensor of shape ``(batch, input_dim)`` and outputs a probability
between 0 and 1 indicating likelihood of fraud.  The internal pipeline is:

1. **Autoencoder** – compresses the input into a latent representation.
2. **Self‑attention** – transforms the latent vector with a parametric attention mechanism.
3. **Quantum‑inspired FCL** – evaluates a simple variational circuit on the transformed
   latent features and returns an expectation value.
4. **Classifier** – combines the two signals and applies a sigmoid.

The design mirrors the structure of the quantum seed, allowing a side‑by‑side
comparison between the two paradigms.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Import the lightweight building blocks that were provided as seed modules.
# They are kept separate to avoid duplicating code and to preserve traceability.
# The exact import paths depend on the repository layout; replace with the
# appropriate relative imports if necessary.
try:
    from Autoencoder import AutoencoderNet, AutoencoderConfig
except Exception:  # pragma: no cover
    # Minimal stubs that mimic the API expected by the hybrid class.
    class AutoencoderConfig:
        def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout

    class AutoencoderNet(nn.Module):
        def __init__(self, config: AutoencoderConfig):
            super().__init__()
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
                decoder_layers.append(nn.Linear(in_dim, hidden))
                decoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.encoder(inputs)

        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            return self.decoder(latents)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.decode(self.encode(inputs))

try:
    from SelfAttention import SelfAttention
except Exception:  # pragma: no cover
    class SelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ torch.as_tensor(inputs, dtype=torch.float32)).numpy()

try:
    from FCL import FCL
except Exception:  # pragma: no cover
    class FCL:
        def __init__(self):
            self.linear = nn.Linear(1, 1)

        def run(self, thetas):
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that chains a classical autoencoder, a
    self‑attention block, and a quantum‑inspired fully‑connected layer before
    producing a binary output.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. Classical auto‑encoder
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        # 2. Self‑attention (classical surrogate of the quantum block)
        self.attention = SelfAttention(embed_dim=latent_dim)
        # 3. Quantum‑inspired fully‑connected layer
        self.fcl = FCL()
        # 4. Final classifier
        self.classifier = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Learnable parameters for the attention mechanism
        self.rotation_params = nn.Parameter(
            torch.randn(latent_dim * 3, dtype=torch.float32)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(latent_dim - 1, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, 1)`` containing probabilities between 0 and 1.
        """
        # Encode → latent representation
        latent = self.autoencoder.encode(inputs)  # shape (batch, latent_dim)

        # Self‑attention transformation (requires numpy arrays)
        rotation = self.rotation_params.detach().cpu().numpy()
        entangle = self.entangle_params.detach().cpu().numpy()
        # Convert latent to numpy for the attention routine
        latent_np = latent.detach().cpu().numpy()
        attention_out = self.attention.run(rotation, entangle, latent_np)
        attention_tensor = torch.as_tensor(attention_out, dtype=torch.float32)

        # Quantum‑inspired FCL expectation value
        fcl_out = self.fcl.run(rotation.tolist())
        fcl_tensor = torch.tensor(fcl_out, dtype=torch.float32)

        # Combine signals
        combined = attention_tensor + fcl_tensor

        logits = self.classifier(combined)
        probs = self.sigmoid(logits)
        return probs

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns a binary prediction."""
        return (self.forward(inputs) > 0.5).float()

__all__ = ["FraudDetectionHybrid"]
