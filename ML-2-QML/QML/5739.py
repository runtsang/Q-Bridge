"""Quantum implementation of ConvGen124.

This module uses torchquantum to encode a 28×28 image into a 4‑wire quantum
device, runs a small variational circuit, measures Pauli‑Z on each wire and
produces a 4‑dimensional feature vector.  The architecture mirrors the
classical version: a quantum encoder followed by a QLayer and a batch‑norm
layer.

The forward method accepts a batch of grayscale images of shape
(batch, 1, 28, 28) and returns a tensor of shape (batch, 4).
"""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class ConvGen124(tq.QuantumModule):
    """
    Quantum counterpart to the classical ConvGen124.
    """

    class QLayer(tq.QuantumModule):
        """
        Small variational layer that applies a random circuit and a few
        parameterised rotations to 4 wires.
        """

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Random layer provides a good starting point for optimisation
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entangling gate
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            # Add a few more single‑qubit gates for expressivity
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder: flatten the 28×28 image to 784 values and then use a
        # 4‑wire parameterised encoder (e.g., 4x4_ryzxy) to embed the data.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 4).
        """
        bsz = x.size(0)
        # Prepare a quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Average‑pool the image to 6×6 and flatten to 16 values
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)

        # Encode the classical data into the quantum state
        self.encoder(qdev, pooled)

        # Apply the variational layer
        self.q_layer(qdev)

        # Measure all qubits in the Z basis
        out = self.measure(qdev)

        # Normalise with batch‑norm
        return self.norm(out)

__all__ = ["ConvGen124"]
