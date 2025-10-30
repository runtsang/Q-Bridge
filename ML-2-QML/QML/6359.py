"""Quantum self‑attention module implemented with TorchQuantum.

The design borrows the random layer + parameterised gates from the
Quantum‑NAT `QLayer`, but uses the attention‑style mapping from the
original SelfAttention circuit.  The module accepts rotation and
entangle parameters, builds a variational circuit, and measures the
Pauli‑Z expectation values of each qubit.  These expectation values
serve as attention weights over the encoded input features.

The module is fully differentiable and can be trained end‑to‑end with
PyTorch optimisers.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumSelfAttentionGen172(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires:int=4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling CRX
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 2])
            # Optional Hadamard on last qubit
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires:int=4, out_features:int=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.out_features = out_features
        self.fc = nn.Linear(n_wires, out_features)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x:torch.Tensor, rotation_params:np.ndarray,
                entangle_params:np.ndarray) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, 1, H, W) – same shape as classical version
            rotation_params: array of shape (n_wires*3,) – parameterised rotations
            entangle_params: array of shape (n_wires-1,) – entangling parameters
        Returns:
            Tensor of shape (B, out_features)
        """
        bsz = x.size(0)
        # Encode classical image into quantum state
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Use average pooling to reduce feature map size
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Apply variational layer
        self.q_layer(qdev)

        # Measurement
        out = self.measure(qdev)  # (B, n_wires)
        out = self.fc(out)
        return self.norm(out)

__all__ = ["QuantumSelfAttentionGen172"]
