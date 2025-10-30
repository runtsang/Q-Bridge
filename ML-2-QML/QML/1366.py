"""Variational quantum circuit with entanglement and parameter‑shiftable gates for 4‑class classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModel(tq.QuantumModule):
    """Hybrid quantum module inspired by Quantum‑NAT, extended with a multi‑layer variational ansatz."""

    class QLayer(tq.QuantumModule):
        """Parameter‑shiftable variational layer with entanglement across all wires."""
        def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Parameters: (layer, wire, gate) where gate ∈ {RX, RY, RZ}
            self.layer_params = nn.Parameter(
                torch.randn(n_layers, n_wires, 3)
            )
            self.entangler = tq.ControlledX(n_wires=n_wires)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in range(self.n_layers):
                for wire in range(self.n_wires):
                    params = self.layer_params[layer, wire]
                    tqf.rx(qdev, params[0], wires=wire, static=self.static_mode, parent_graph=self.graph)
                    tqf.ry(qdev, params[1], wires=wire, static=self.static_mode, parent_graph=self.graph)
                    tqf.rz(qdev, params[2], wires=wire, static=self.static_mode, parent_graph=self.graph)
                # Entangle adjacent qubits and wrap‑around
                for i in range(0, self.n_wires - 1, 2):
                    self.entangler(qdev, wires=[i, i + 1])
                self.entangler(qdev, wires=[self.n_wires - 1, 0])

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Simple 2‑D pooling to match the encoder dimensionality
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QFCModel"]
