"""Hybrid sampler‑regressor implemented with a quantum circuit.

The module defines:
- SamplerQNNGen048: a torchquantum module that encodes classical inputs, applies a random layer and
  parameterized rotations, measures all qubits, and produces both a 2‑class probability distribution
  and a regression output.
- SamplerQNN: factory function for backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

__all__ = ["SamplerQNNGen048", "SamplerQNN"]


class SamplerQNNGen048(tq.QuantumModule):
    """Quantum hybrid sampler‑regressor."""

    class QLayer(tq.QuantumModule):
        """Random layer + trainable rotations."""

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

    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # Classical encoding via Ry rotations
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Heads
        self.sampler_head = nn.Linear(num_wires, 2)
        self.regression_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of classical feature vectors of shape (batch, num_wires).

        Returns
        -------
        distribution : torch.Tensor
            Softmax probability over two outcomes.
        regression : torch.Tensor
            Scalar regression prediction.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        # Convert Pauli‑Z expectations to probabilities for the first qubit
        probs = (1 + features[:, 0]) / 2  # shape (bsz,)
        distribution = torch.stack([probs, 1 - probs], dim=-1)  # shape (bsz, 2)
        regression = self.regression_head(features).squeeze(-1)
        return distribution, regression


def SamplerQNN() -> SamplerQNNGen048:
    """Factory function for backward compatibility."""
    return SamplerQNNGen048()
