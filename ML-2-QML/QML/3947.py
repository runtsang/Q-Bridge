"""Hybrid quantum self‑attention module built with torchquantum.

The quantum circuit encodes the input features with a GeneralEncoder, applies a
parameterised quantum layer and measures Pauli‑Z to obtain attention scores.
The structure is inspired by the QFCModel and the original SelfAttention circuit.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class HybridSelfAttentionQML(tq.QuantumModule):
    """
    Quantum self‑attention module for differentiable training.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical data to quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
            Only the first `n_wires` features are used for encoding.

        Returns
        -------
        torch.Tensor
            Attention‑like output of shape (batch, seq_len, n_wires).
        """
        bsz, seq_len, _ = x.shape
        # Flatten per position to feed into encoder
        encoded = x.contiguous().view(bsz * seq_len, -1)
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz * seq_len, device=x.device, record_op=True)
        # Encode classical data
        self.encoder(qdev, encoded)
        # Apply quantum layer
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out).view(bsz, seq_len, self.n_wires)

def SelfAttention():
    """
    Public factory that mirrors the original interface.
    Returns an instance of :class:`HybridSelfAttentionQML`.
    """
    return HybridSelfAttentionQML(n_wires=4)
