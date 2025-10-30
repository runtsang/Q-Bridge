"""Quantum module with a parameter‑shared hybrid variational ansatz
and measurement‑based readout for Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Hybrid variational circuit that extends the seed QFCModel.

    The new design uses a *parameter‑shared* ansatz across all
    four qubits.  Each layer consists of a single‑qubit rotation
    (RX, RY, RZ) followed by a two‑qubit controlled‑rotation
    (CRX).  The circuit is repeated for ``n_layers`` and the
    parameters are shared across the layers, resulting in a
    compact yet expressive model.

    The measurement stage uses a *Pauli‑Z* readout for each qubit
    and then *classical post‑processing* (a linear layer) to map
    the expectation values to the final 4‑dimensional output.

    The design is intentionally lightweight to enable
    large‑scale training on a CPU or a GPU‑enabled simulator.
    """

    class SharedAnsatz(tq.QuantumModule):
        """Parameter‑shared ansatz block."""
        def __init__(self, n_wires: int, n_layers: int):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Rotation gates (shared parameters)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Controlled rotation (shared)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for _ in range(self.n_layers):
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                    self.rz(qdev, wires=wire)
                # Controlled rotation between wire pairs (0,2) and (1,3)
                self.crx(qdev, wires=[0, 2])
                self.crx(qdev, wires=[1, 3])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.ansatz = self.SharedAnsatz(self.n_wires, n_layers=3)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Global average pooling to match the seed
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into qubit states
        self.encoder(qdev, pooled)
        # Apply the shared ansatz
        self.ansatz(qdev)
        # Readout
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
