"""Quantum hybrid sampler network.

   Implements a quantum variational sampler that accepts a 2‑D image
   as input, encodes it into a 4‑qubit device using a general RyZXY
   encoder, applies a stochastic QLayer (RandomLayer + trainable
   single‑qubit rotations and controlled‑rotations), and finally
   measures all qubits in the Pauli‑Z basis.  The resulting expectation
   values are normalised to a probability distribution over 4 outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridSamplerQNN(tq.QuantumModule):
    """Quantum sampler network with classical feature extraction.

    The network mirrors the classical HybridSamplerQNN but replaces
    the fully‑connected head with a variational quantum circuit.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer consisting of a random circuit and trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a flattened image patch into a 4‑qubit state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over four outputs.

        Parameters
        ----------
        x
            Input image tensor of shape (batch, 1, H, W).  The network
            first averages over a 6×6 window to produce a 16‑dimensional
            feature vector that is fed to the encoder.
        """
        bsz = x.shape[0]
        # Reduce spatial resolution to a 16‑dim vector similar to QFCModel
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        # Convert expectation values (-1,1) to probabilities (0,1)
        probs = (out + 1) / 2
        return probs
