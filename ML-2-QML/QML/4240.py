"""Quantum‑enhanced hybrid block for the UnifiedQuantumRegressor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumHybridBlock(tq.QuantumModule):
    """
    Variational circuit that can act as the hybrid block in UnifiedQuantumRegressor.
    The circuit uses a parameter‑efficient encoding, a small random layer,
    and a measurement head that returns a feature vector.
    The block also optionally supports a self‑attention‑style entanglement layer
    or a qubit‑reduced LSTM gate pattern.
    """
    def __init__(
        self,
        num_wires: int,
        *,
        encoder_name: str | None = None,
        use_attention_entanglement: bool = False,
        use_lstm_pattern: bool = False,
    ):
        super().__init__()
        self.num_wires = num_wires
        self.use_attention = use_attention_entanglement
        self.use_lstm = use_lstm_pattern

        # Parameter‑efficient encoding: a linear map from classical state to angles
        # Use the built‑in encoder for Ry rotations if available
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict.get(encoder_name or f"{num_wires}xRy", [])
        )

        # Random layer to introduce expressivity
        self.random_layer = tq.RandomLayer(
            n_ops=20,
            wires=list(range(num_wires)),
            use_random=True,
        )

        # Optional entanglement patterns
        if self.use_attention:
            # Controlled‑rotation pattern mimicking self‑attention
            self.entangle = tq.CRX(has_params=True, trainable=True)
        elif self.use_lstm:
            # Reduced CNOT pattern similar to the quantum LSTM
            self.cnot_pattern = nn.ModuleList(
                [
                    tq.CNOT(has_params=False, trainable=False)
                    for _ in range(num_wires)
                ]
            )
        else:
            self.entangle = None

        # Measurement head
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Head to map measurement results to a feature vector
        self.out_dim = num_wires
        self.head = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Shape (batch, num_wires) – classical states to encode.

        Returns
        -------
        torch.Tensor
            Shape (batch, out_dim) – hybrid features.
        """
        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=states.device)

        # Encode classical data
        self.encoder(qdev, states)

        # Random layer
        self.random_layer(qdev)

        # Optional entanglement
        if self.use_attention:
            # Apply a controlled‑rotation between adjacent qubits
            for i in range(self.num_wires - 1):
                self.entangle(qdev, wires=[i, i + 1])
        elif self.use_lstm:
            # Apply a reduced CNOT pattern
            for i, gate in enumerate(self.cnot_pattern):
                if i < self.num_wires - 1:
                    gate(qdev, wires=[i, i + 1])
                else:
                    gate(qdev, wires=[i, 0])

        # Measure
        features = self.measure(qdev)

        # Linear head to produce the final hybrid feature vector
        return self.head(features).squeeze(-1)

__all__ = ["QuantumHybridBlock"]
