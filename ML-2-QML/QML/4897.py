import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum Quanvolution filter (reference 3 quantum side)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encode each pixel value with a separate Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # [B, 4*14*14]

# --------------------------------------------------------------------------- #
# Quantum self‑attention module (inspired by reference 4)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """
    Simple quantum block that mimics a self‑attention style transformation
    by applying a random rotation layer followed by a measurement.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, qdev: tq.QuantumDevice, inputs: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, inputs)
        self.random_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# Final quantum classifier (reference 1 quantum side)
# --------------------------------------------------------------------------- #
class QFCModel(tq.QuantumModule):
    """Quantum fully connected model inspired by the Quantum‑NAT paper."""

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
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# Hybrid quantum architecture
# --------------------------------------------------------------------------- #
class HybridNATQuantumModel(tq.QuantumModule):
    """
    Quantum counterpart of the classical HybridNATModel.  The pipeline is:

        1. Quanvolution filter (2×2 patches → 4‑qubit kernels)
        2. Quantum self‑attention block that further mixes the patch embeddings
        3. Final linear classifier mapping the 4‑dimensional representation to logits
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = QuantumSelfAttention()
        self.classifier = nn.Linear(4, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution
        features = self.qfilter(x)                     # [B, 4*14*14]
        # 2. Reduce spatial dimension: average over patches to obtain 4‑dim vector
        reshaped = features.view(x.size(0), 4, 14, 14)
        pooled = reshaped.mean(dim=(2, 3))              # [B, 4]
        # 3. Quantum self‑attention
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device)
        attn_out = self.attention(qdev, pooled)         # [B, 4]
        # 4. Classify
        logits = self.classifier(attn_out)              # [B, num_classes]
        return self.softmax(logits)

__all__ = ["HybridNATQuantumModel"]
