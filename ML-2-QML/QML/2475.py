from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridSamplerQNN(tq.QuantumModule):
    """
    Quantum hybrid sampler that encodes image features into a 4‑qubit
    variational circuit and produces a 2‑class probability distribution.
    The architecture integrates the GeneralEncoder from Quantum‑NAT
    with a random layer and custom gates, mirroring the classical CNN‑FC
    backbone while adding quantum expressivity.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
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
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability distribution over 2 classes, shape (batch, 2).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        # Encode image features into the quantum device
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Apply the variational layer
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)  # shape (batch, 4)
        # Use first two qubits as logits for a 2‑class softmax
        logits = out[:, :2]
        probs = F.softmax(logits, dim=-1)
        probs = self.norm(probs)
        return probs


def SamplerQNN() -> HybridSamplerQNN:
    """
    Compatibility wrapper that mirrors the original SamplerQNN function.
    Returns an instance of the new HybridSamplerQNN quantum class.
    """
    return HybridSamplerQNN()


__all__ = ["HybridSamplerQNN", "SamplerQNN"]
