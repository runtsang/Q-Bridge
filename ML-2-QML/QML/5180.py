"""QuantumHybridNAT: quantum‑enhanced hybrid architecture.

This implementation mirrors the classical version but replaces the variational
block with a true quantum circuit.  The quantum encoder uses a 4‑wire
`GeneralEncoder` with a 4x4 RY‑Z‑XY pattern, followed by a `QLayer` that
combines random rotations and controlled gates.  The measurement outputs a
4‑dim vector that is added residually to the classical embedding before
classification.

The class is fully compatible with the original `QuantumNAT` API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict


class QuantumHybridNAT(tq.QuantumModule):
    """Quantum‑enabled hybrid network that extends the classical design.

    The architecture follows the same high‑level flow as the classical
    counterpart but injects a quantum variational circuit to process the
    4‑dim embedding produced by the CNN backbone.  Residual scaling and
    shift are preserved, and the final logits are computed by a linear
    classifier.
    """

    def __init__(self, *, num_classes: int = 4, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # CNN backbone (identical to the classical version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Projection to 4‑dim vector
        self.proj = nn.Linear(16 * 7 * 7, 4)

        # Residual scaling and shift
        self.res_scale = nn.Parameter(torch.ones(4))
        self.res_shift = nn.Parameter(torch.zeros(4))

        # Quantum encoder
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict["4x4_ryzxy"]
        )

        # Variational quantum circuit (QLayer)
        self.q_layer = self.QLayer()

        # Batch‑norm
        self.batch_norm = nn.BatchNorm1d(4)

        # Final classifier
        self.out = nn.Linear(4, num_classes)

    class QLayer(tq.QuantumModule):
        """Small quantum variational block used in the hybrid model."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(4))
            )
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])
            return self.measure(qdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum processing.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        # Classical feature extraction
        feats = self.features(x)

        # 4‑dim embedding
        embed = self.proj(feats)

        # Residual connection
        residual = self.res_scale * embed + self.res_shift

        # Quantum device
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=bsz,
            device=x.device,
            record_op=False,
        )

        # Encode the residual into the quantum device
        self.encoder(qdev, residual)

        # Apply variational circuit
        qout = self.q_layer(qdev)

        # Combine quantum output with residual
        combined = qout + residual

        # Normalise and classify
        normed = self.batch_norm(combined)
        logits = self.out(normed)

        return F.log_softmax(logits, dim=1)


__all__ = ["QuantumHybridNAT"]
