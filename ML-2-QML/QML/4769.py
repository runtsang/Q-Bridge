"""Quantum implementation of the Hybrid NAT model.

The quantum version mirrors the classical architecture by providing a quantum encoder, a random layer, a quantum sampler network, and a quantum filter that together generate a 7‑dimensional feature vector.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F


class QuantumFilter(tq.QuantumModule):
    """Simple 1‑qubit filter that maps an input scalar to a Z‑measurement."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 1
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Encode the scalar as an RX rotation
        qdev.rx(x.squeeze(-1), 0)
        out = self.measure(qdev)
        # Return expectation value of Z (mean over shots)
        return out.mean(dim=1, keepdim=True)


class QuantumSampler(tq.QuantumModule):
    """Parameter‑ised quantum sampler that outputs a 2‑dimensional probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz = inputs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=inputs.device)
        qdev.rx(inputs[:, 0], 0)
        qdev.rx(inputs[:, 1], 1)
        qdev.cx(0, 1)
        out = self.measure(qdev)
        # Convert measurement counts to a 2‑dimensional probability vector
        probs = out.mean(dim=0).unsqueeze(0).expand(bsz, -1)
        return probs


class QuantumNAT(tq.QuantumModule):
    """Quantum counterpart of the Hybrid NAT model."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.filter = QuantumFilter()
        self.sampler = QuantumSampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Create a quantum device for the 4‑wire block
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Pool the image and encode into the qubits
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Apply the random layer
        self.q_layer(qdev)
        out_q = self.measure(qdev)                     # (B,4)
        # Quantum filter on a scalar derived from the image
        scalar = x.mean(dim=[1, 2, 3], keepdim=True)   # (B,1)
        filt_out = self.filter(scalar)                 # (B,1)
        # Quantum sampler on a 2‑dimensional slice of the pooled tensor
        samp_out = self.sampler(pooled[:, :2])          # (B,2)
        # Concatenate all components
        concatenated = torch.cat([out_q, samp_out, filt_out], dim=1)
        return self.norm(concatenated)


__all__ = ["QuantumNAT"]
