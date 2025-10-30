import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Autoencoder – lightweight feature extractor used by the classical block
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder used as a feature encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

def Autoencoder(input_dim: int, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    """Factory mirroring the quantum helper."""
    return AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)

# --------------------------------------------------------------------------- #
# 2. Fraud‑detection inspired linear layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def build_fraud_layer(params: FraudLayerParameters, clip: bool = False) -> nn.Linear:
    """Create a 2×2 linear layer from fraud‑detection parameters."""
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias   = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    return linear

# --------------------------------------------------------------------------- #
# 3. Classical Self‑attention with graph‑based gating
# --------------------------------------------------------------------------- #
class SelfAttention:
    """
    Classical self‑attention that embeds inputs with an autoencoder,
    applies fraud‑layer derived weights as query/key projections, and
    uses a fidelity‑based graph to gate the output.
    """
    def __init__(self, embed_dim: int, latent_dim: int = 32):
        self.embed_dim = embed_dim
        self.autoencoder = Autoencoder(embed_dim, latent_dim=latent_dim)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters reshaped to (embed_dim, embed_dim) for the query projection.
        entangle_params : np.ndarray
            Parameters reshaped to (embed_dim, embed_dim) for the key projection.
        inputs : np.ndarray
            Batch of input vectors of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention weighted outputs of shape (batch, embed_dim).
        """
        # Encode with the autoencoder
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        encoded = self.autoencoder.encode(inp)

        # Build query/key projections
        q_w = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k_w = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        query = encoded @ q_w
        key   = encoded @ k_w

        # Attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)

        # Value is the encoded representation
        value = encoded
        output = scores @ value

        # --------------------------------------------------------------------- #
        # 4. Fidelity‑based adjacency graph to gate the output
        # --------------------------------------------------------------------- #
        states = output.detach().cpu()
        graph = nx.Graph()
        graph.add_nodes_from(range(states.size(0)))
        for i in range(states.size(0)):
            for j in range(i + 1, states.size(0)):
                fid = torch.dot(states[i], states[j]).item()
                if fid >= 0.8:            # hard threshold
                    graph.add_edge(i, j, weight=1.0)

        # Create a gating mask: average of neighbours
        mask = torch.zeros_like(output)
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                mask[node] = torch.mean(output[neighbors], dim=0)
        gated_output = output * mask

        return gated_output.numpy()

def SelfAttention() -> SelfAttention:
    """Factory that mirrors the quantum helper."""
    return SelfAttention(embed_dim=4)

__all__ = ["SelfAttention", "Autoencoder", "AutoencoderNet", "FraudLayerParameters", "build_fraud_layer"]
