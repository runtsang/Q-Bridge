import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------------
# Auto‑encoder components (from Autoencoder.py)
# ------------------------------------------------------------------
class AutoencoderConfig:
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
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

# ------------------------------------------------------------------
# Hybrid fully‑connected layer
# ------------------------------------------------------------------
class HybridFCL(nn.Module):
    """
    A classical hybrid layer that emulates a quantum FCL block with
    explicit self‑attention and auto‑encoder stages.
    """

    def __init__(
        self,
        n_features: int = 1,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # Shapes for the attention parameters
        self.attn_rot_shape = (3 * n_qubits, 1)
        self.attn_ent_shape = (n_qubits - 1, 1)

        # Auto‑encoder backbone
        config = AutoencoderConfig(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(config)

        # Final projection
        self.linear = nn.Linear(n_features, 1)

    def _attention(self, inputs: torch.Tensor, rotation: torch.Tensor, entangle: torch.Tensor) -> torch.Tensor:
        """Classical self‑attention computation."""
        Q = inputs @ rotation.reshape(self.n_qubits, -1)
        K = inputs @ entangle.reshape(self.n_qubits, -1)
        V = inputs
        scores = torch.softmax(Q @ K.T / np.sqrt(self.n_qubits), dim=-1)
        return scores @ V

    def run(self, thetas: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Run the hybrid layer.

        Parameters
        ----------
        thetas : array‑like
            Concatenated rotation (3*n_qubits) and entanglement (n_qubits‑1) parameters.
        inputs : array‑like
            Input feature matrix (batch_size × n_features).

        Returns
        -------
        np.ndarray
            Output of the hybrid FCL, shape (1,).
        """
        rotation = torch.as_tensor(thetas[:3 * self.n_qubits], dtype=torch.float32)
        entangle = torch.as_tensor(thetas[3 * self.n_qubits :], dtype=torch.float32)

        inp_tensor = torch.as_tensor(inputs, dtype=torch.float32)

        # Self‑attention
        attn_out = self._attention(inp_tensor, rotation, entangle)

        # Auto‑encoder bottleneck
        latent = self.autoencoder.encode(attn_out)
        decoded = self.autoencoder.decode(latent)

        # Final projection
        out = torch.tanh(self.linear(decoded)).mean(dim=0)
        return out.detach().numpy()

__all__ = ["HybridFCL"]
