from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# ----------------------------------------------------------------------
# Imports from the seed modules
# ----------------------------------------------------------------------
from.GraphQNN import random_network, feedforward, fidelity_adjacency
from.Autoencoder import Autoencoder
from.QLSTM import LSTMTagger
from.QCNN import QCNN

# ----------------------------------------------------------------------
# Classical hybrid model
# ----------------------------------------------------------------------
class SharedClassName(nn.Module):
    """
    Hybrid model that combines a graph‑based feed‑forward network,
    an autoencoder, a sequence LSTM tagger and a convolutional head.
    The network is fully differentiable and can be trained end‑to‑end
    with standard PyTorch optimizers.
    """

    def __init__(
        self,
        graph_arch: list[int],
        autoenc_cfg: dict,
        lstm_cfg: dict,
        cnn_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        # 1️⃣  Graph QNN (classical)
        self.graph_arch, self.graph_units, _, self.graph_target = random_network(
            graph_arch, samples=10
        )

        # 2️⃣  Autoencoder
        self.autoencoder = Autoencoder(**autoenc_cfg)

        # 3️⃣  Sequence LSTM tagger
        self.lstm = LSTMTagger(**lstm_cfg)

        # 4️⃣  Convolutional head
        self.cnn = QCNN()

        # 5️⃣  Adjacency graph of latent states (for analysis)
        latent_samples = self._sample_latent(50)
        self.adjacency = fidelity_adjacency(
            latent_samples, threshold=0.9, secondary=0.8, secondary_weight=0.2
        )

    # ------------------------------------------------------------------
    # Helper: generate random latent vectors for adjacency graph
    # ------------------------------------------------------------------
    def _sample_latent(self, n: int) -> torch.Tensor:
        """
        Create `n` latent vectors by feeding random inputs through the
        autoencoder encoder.  These vectors are later used to build the
        fidelity‑based adjacency graph.
        """
        dummy_inputs = torch.randn(n, self.autoencoder.encoder[-1].in_features)
        latents = self.autoencoder.encode(dummy_inputs)
        return latents

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input shape: (batch, seq_len, feature_dim)
        1. Autoencode each time step → latent representation.
        2. Feed latent sequence into LSTM tagger → hidden states.
        3. Average LSTM outputs across the sequence.
        4. Classify with the convolutional head.
        """
        batch, seq_len, _ = x.shape

        # Autoencoder latent
        latent = self.autoencoder.encode(x.view(-1, x.size(-1)))
        latent = latent.view(batch, seq_len, -1)

        # LSTM tagger
        lstm_out, _ = self.lstm(latent)  # (batch, seq_len, hidden_dim)

        # Pooling across the sequence
        pooled = lstm_out.mean(dim=1)  # (batch, hidden_dim)

        # Convolutional classification
        out = self.cnn(pooled)  # (batch, 1)
        return out


__all__ = ["SharedClassName"]
