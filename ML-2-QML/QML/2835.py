"""Hybrid quantum classifier with CNN feature extractor and variational layer.

The quantum module reuses the same feature extraction pipeline as the
classical counterpart and replaces the final linear head with a
parameterised quantum circuit.  The circuit is built using
torchquantum and follows the incremental data‑uploading pattern
from the first reference pair.  It outputs expectation values of
Pauli‑Z on each qubit, which are then linearly combined to produce
binary logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumClassifier(tq.QuantumModule):
    """CNN + variational quantum circuit for binary classification.

    The structure mirrors the classical `HybridQuantumClassifier` but
    replaces the fully‑connected head with a quantum layer that operates
    on 4 qubits.  The encoder uses a deterministic 4×4 RYZXY pattern
    to map the pooled image features into the quantum state.  A
    lightweight variational block (random layer + single‑qubit gates
    + CNOTs) follows, and the expectation values of Pauli‑Z are measured.
    Finally, a classical linear layer maps the 4‑dimensional quantum
    output to 2 logits.
    """

    class QLayer(tq.QuantumModule):
        """Variational block inspired by the QFCModel seed."""

        def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.depth = depth

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            """Apply the variational block."""
            # Random layer provides a fixed, expressive starting point.
            self.random_layer(qdev)
            # Repeat a simple single‑qubit rotation pattern.
            for _ in range(self.depth):
                for w in range(self.n_wires):
                    self.rx0(qdev, wires=w)
                    self.ry0(qdev, wires=w)
                    self.rz0(qdev, wires=w)
                # Entangle neighbouring qubits with CNOTs.
                for w in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[w, w + 1])
            # Add a final Hadamard to create superposition.
            tqf.hadamard(qdev, wires=list(range(self.n_wires)))

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        # Classical feature extractor identical to the ML version.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Encoder that maps the pooled features into qubit rotations.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires, depth=2)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head mapping 4 expectation values to logits.
        self.fc = nn.Linear(self.n_wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        feats = self.features(x)
        # Average‑pool to 4×4 feature map to match encoder size.
        pooled = F.avg_pool2d(feats, kernel_size=feats.shape[2] // 4)
        flat = pooled.view(bsz, -1)
        # Build a quantum device for each batch element.
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical data into the quantum state.
        self.encoder(qdev, flat)
        # Variational processing.
        self.q_layer(qdev)
        # Measurement of Pauli‑Z expectation values.
        q_out = self.measure(qdev)  # shape (bsz, n_wires)
        logits = self.fc(q_out)
        return self.norm(logits)


__all__ = ["HybridQuantumClassifier"]
