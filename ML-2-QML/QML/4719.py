"""Quantum hybrid network mirroring the classical architecture.

It replaces the 2×2 patch convolution with a quantum kernel based on a
randomised two‑qubit circuit, and the final fully‑connected head with a
Quantum‑NAT inspired QFC layer.  The module supports both classification
and regression via a ``regression`` flag.  When needed, the same logic
can be wrapped in a Qiskit EstimatorQNN for backend‑agnostic execution.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum 2×2 patch filter using a random layer and Pauli‑Z measurement."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)  # [B, 4*14*14]

class QFCQuantumLayer(tq.QuantumModule):
    """Quantum fully‑connected layer inspired by Quantum‑NAT."""
    def __init__(self) -> None:
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

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum hybrid network: quanvolution filter + QFC layer + linear head."""
    def __init__(self, num_classes: int = 10, regression: bool = False) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.qfc = QFCQuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.qfilter.n_wires)
        self.classifier = nn.Linear(self.qfilter.n_wires, num_classes if not regression else 1)
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.qfilter.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode the classical patches into qubits
        patches = self.qfilter(x)  # [B, 4*14*14]
        # Collapse patches to a single vector per sample for the QFC layer
        encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        encoder(qdev, patches)
        self.qfc(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        logits = self.classifier(out)
        return logits if self.regression else F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilterQuantum", "QFCQuantumLayer"]
