import torch
import torch.nn as nn
import networkx as nx
from.Autoencoder import Autoencoder
from.GraphQNN import fidelity_adjacency

class QuantumHybridNAT(nn.Module):
    """Hybrid classical encoder that extracts image features, compresses them via an autoencoder,
    and builds a similarity graph of the latent vectors.  The resulting latent representation
    can be fed into a quantum variational network that implements a QCNN‑style ansatz.

    This class combines the CNN backbone from QuantumNAT, the autoencoder from Autoencoder.py,
    and the graph‑based similarity construction from GraphQNN.py into a single, end‑to‑end
    classical module.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 32, graph_threshold: float = 0.8):
        super().__init__()
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Autoencoder for latent compression
        self.autoencoder = Autoencoder(
            input_dim=16 * 7 * 7,
            latent_dim=latent_dim,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | nx.Graph]:
        """Return a dictionary with the latent vector and a weighted adjacency graph.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        dict
            {'latent': Tensor, 'adjacency': nx.Graph}
        """
        # Extract convolutional features
        feats = self.features(x)  # (B, 16, 7, 7)
        flat = feats.view(feats.size(0), -1)  # (B, 784)
        # Encode to latent space
        latent = self.autoencoder.encode(flat)  # (B, latent_dim)
        # Build similarity graph of latent vectors
        adjacency = fidelity_adjacency(
            [latent[i] for i in range(latent.size(0))],
            threshold=self.graph_threshold,
        )
        return {"latent": latent, "adjacency": adjacency}

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns only the latent representation."""
        with torch.no_grad():
            return self.forward(x)["latent"]
