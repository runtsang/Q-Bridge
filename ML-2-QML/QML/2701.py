"""Hybrid quantum model that mirrors the classical architecture.

The quantum version replaces the fully‑connected head with a variational circuit that processes averaged patch measurements.  
A 2×2 patch encoder, a RandomLayer, and a measurement stage form the quanvolution block; the aggregated features are encoded into 4 qubits and processed by a multi‑gate variational circuit before a linear classifier outputs class logits."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Variational layer that adds expressivity to the quantum circuit."""

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
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])


class HybridNatQuantumModel(tq.QuantumModule):
    """
    Quantum counterpart of the classical HybridNatQuantumModel.

    Architecture:
        1. AvgPool2d(6) → 14×14 feature map.
        2. For each 2×2 patch:
            - Encode with 4 Ry gates (one per pixel).
            - Apply RandomLayer(8 ops).
            - Measure all qubits (Pauli‑Z).
        3. Average the 4‑dimensional patch measurements → 4‑dim vector.
        4. Encode this vector into 4 qubits via Ry gates.
        5. Apply QLayer (variational circuit).
        6. Measure all qubits → 4‑dim output.
        7. BatchNorm1d(4) → Linear(4, 10) → LogSoftmax.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Patch encoder: 4 qubits, one Ry per pixel
        self.patch_encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.patch_random = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.patch_measure = tq.MeasureAll(tq.PauliZ)

        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.classifier = nn.Linear(self.n_wires, 10)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device

        # 1. Reduce spatial resolution
        pooled = F.avg_pool2d(x, 6).view(bsz, 28, 28)

        # 2. Process each 2×2 patch
        patch_meas = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Shape: (bsz, 4)
                data = torch.stack(
                    [
                        pooled[:, r, c],
                        pooled[:, r, c + 1],
                        pooled[:, r + 1, c],
                        pooled[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.patch_encoder(qdev, data)
                self.patch_random(qdev)
                meas = self.patch_measure(qdev).view(bsz, self.n_wires)
                patch_meas.append(meas)

        # 3. Average across 14×14 patches → (bsz, 4)
        patch_meas = torch.stack(patch_meas, dim=1)  # (bsz, 196, 4)
        features = patch_meas.mean(dim=1)  # (bsz, 4)

        # 4. Encode aggregated features into qubits
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        for i in range(self.n_wires):
            qdev.apply(tq.RY, wires=[i], params=features[:, i])

        # 5. Variational circuit
        self.q_layer(qdev)

        # 6. Measurement
        out_q = self.measure(qdev).view(bsz, self.n_wires)
        out_q = self.norm(out_q)

        # 7. Classical classifier
        logits = self.classifier(out_q)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridNatQuantumModel"]
