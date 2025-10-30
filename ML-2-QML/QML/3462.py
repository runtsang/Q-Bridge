"""Hybrid SamplerQNN with a classical CNN backbone and a variational quantum head.
The architecture combines ideas from the 2‑qubit SamplerQNN and the 4‑qubit
Quantum‑NAT QFCModel.  The CNN extracts features, which are encoded into a
4‑qubit state that is then processed by a random layer and a small
variational block.  The resulting state vector is converted into a 4‑class
probability distribution, and a `sample` method uses the state‑vector to
draw categorical samples."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class SamplerQNN(tq.QuantumModule):
    """
    Hybrid classical–quantum sampler network.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational block consisting of a random layer and a small set of
        trainable single‑ and two‑qubit gates.  Inspired by the seed
        implementation in Quantum‑NAT.
        """
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder maps a 16‑dim feature vector to a 4‑qubit state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a 4‑class probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability distribution over four classes (shape: (B, 4)).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        # Classical preprocessing (average‑pooling as in the seed)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        # Retrieve the state vector and compute joint probabilities
        state = qdev.get_state()  # shape: (bsz, 2**4)
        probs = torch.abs(state)**2
        probs = probs.reshape(bsz, 2, 2, 2, 2)          # (bsz, q0, q1, q2, q3)
        # Sum over the last two qubits to obtain a 4‑class distribution
        probs = probs.sum(dim=(3, 4)).reshape(bsz, 4)
        # Ensure numerical stability
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        quantum output.  This method uses the state‑vector probabilities
        and `torch.multinomial` for efficient sampling.

        Parameters
        ----------
        x : torch.Tensor
            Input images.
        num_samples : int
            Number of independent draws per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (B, num_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)

__all__ = ["SamplerQNN"]
