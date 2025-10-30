from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

class QuantumLatentTransformer(tq.QuantumModule):
    """
    Variational quantum circuit that transforms a classical latent vector.
    Encodes the vector into a quantum state, applies a trainable layer,
    and measures to produce a new latent representation.
    """
    def __init__(self, latent_dim: int, n_wires: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_wires = n_wires
        # Encoder that maps classical features to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        # Variational layer
        self.q_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Projection back to classical latent space
        self.head = nn.Linear(n_wires, latent_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Tensor of shape (batch, latent_dim)
        Returns:
            Tensor of shape (batch, latent_dim) after quantum processing.
        """
        bsz = latent.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=latent.device)
        self.encoder(qdev, latent)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features)

__all__ = ["QuantumLatentTransformer"]
