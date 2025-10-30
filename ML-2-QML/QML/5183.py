from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import cnot
# Re‑use the data generation and dataset class from the original anchor
from.QuantumRegression import generate_superposition_data, RegressionDataset

class HybridQuantumAttention(tq.QuantumModule):
    """Quantum self‑attention block: parameterised rotations followed by a CNOT ladder."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Encode each input wire with a tunable RX
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.rot = [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Apply the rotation parameters
        for wire, gate in enumerate(self.rot):
            gate(qdev, wires=wire)
        # Entangle neighbouring qubits
        for wire in range(self.n_wires - 1):
            cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that couples a feature encoder, a random layer,
    a quantum attention sub‑module, and a linear read‑out head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Feature encoder (Ry on each wire)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational random layer
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        # Quantum attention sub‑module
        self.attn = HybridQuantumAttention(num_wires)
        # Final measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of quantum states of shape (batch, 2**num_wires).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode the input states
        self.encoder(qdev, state_batch)
        # Variational processing
        self.q_layer(qdev)
        # Quantum attention
        attn_feat = self.attn(qdev)
        # Measurement of all qubits
        features = self.measure(qdev) + attn_feat
        # Linear read‑out
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
