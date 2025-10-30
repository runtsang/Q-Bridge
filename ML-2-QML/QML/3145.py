"""Hybrid quantum model with quantum encoder and feed‑forward head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

__all__ = ["HybridQuantumNAT"]

class HybridQuantumNAT(tq.QuantumModule):
    """Hybrid model that combines a classical CNN backbone with quantum
    encoding and a quantum feed‑forward head.  The downstream classifier
    is a simple linear layer for demonstration purposes.
    """

    # --------------------------------------------------------------------------- #
    #  Classical CNN‑FC backbone (unchanged from the classical seed)
    # --------------------------------------------------------------------------- #
    class _CNNBackbone(nn.Module):
        def __init__(self, in_channels: int = 1) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.features(x)
            flat = feat.view(x.shape[0], -1)
            out = self.fc(flat)
            return self.norm(out)

    # --------------------------------------------------------------------------- #
    #  Quantum encoder for the CNN features
    # --------------------------------------------------------------------------- #
    class _QuantumEncoder(tq.QuantumModule):
        def __init__(self, n_qubits: int = 4) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode the classical vector into a quantum state and return
            expectation values.  Input shape: (batch, 4).  Output shape: (batch, 4)."""
            qdev = tq.QuantumDevice(
                n_wires=self.n_qubits,
                bsz=x.shape[0],
                device=x.device,
                record_op=True,
            )
            self.encoder(qdev, x)
            return self.measure(qdev)

    # --------------------------------------------------------------------------- #
    #  Quantum feed‑forward head
    # --------------------------------------------------------------------------- #
    class _QFFHead(tq.QuantumModule):
        def __init__(self, n_qubits: int = 4, hidden_dim: int = 32, out_features: int = 2) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            # Encoder mapping the 4‑dimensional input to a quantum state
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            # Parameterised Ry gates on each qubit
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
            # Classical post‑processing
            self.linear1 = nn.Linear(n_qubits, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            qdev = tq.QuantumDevice(
                n_wires=self.n_qubits,
                bsz=batch,
                device=x.device,
                record_op=True,
            )
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            q_out = self.measure(qdev)
            return self.linear2(F.relu(self.linear1(q_out)))

    # --------------------------------------------------------------------------- #
    #  Simple linear classifier (classical)
    # --------------------------------------------------------------------------- #
    class _Classifier(nn.Module):
        def __init__(self, in_features: int, num_classes: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def __init__(self, num_classes: int = 2, use_qff: bool = True) -> None:
        super().__init__()
        self.backbone = self._CNNBackbone()
        self.encoder = self._QuantumEncoder()
        self.qff_head = self._QFFHead() if use_qff else nn.Identity()
        self.classifier = self._Classifier(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (batch, 4)
        qfeat = self.encoder(feat)       # (batch, 4)
        if isinstance(self.qff_head, nn.Identity):
            out = self.classifier(qfeat)
        else:
            out = self.classifier(self.qff_head(qfeat))
        return out
