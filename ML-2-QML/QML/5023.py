from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
from Autoencoder import Autoencoder

class HybridRegression(tq.QuantumModule):
    """Quantum counterpart of HybridRegression.

    The model uses a classical auto‑encoder, a quantum encoder,
    a variational layer, measurement, and a linear regression head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=num_features,
            latent_dim=8,
            hidden_dims=(32,),
            dropout=0.0,
        )
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        encoded = self.autoencoder.encode(state_batch)          # (B, latent_dim)
        encoded_flat = encoded.view(bsz, -1)                    # (B, latent_dim)
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                bsz=bsz,
                                device=state_batch.device)
        self.encoder(qdev, encoded_flat)
        self.q_layer(qdev)
        features = self.measure(qdev)                           # (B, num_wires)
        return self.head(features).squeeze(-1)
