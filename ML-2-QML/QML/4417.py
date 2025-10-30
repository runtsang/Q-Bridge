import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum implementation of the 2×2 patch encoder used in the classical quanvolution filter."""
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
        return torch.cat(patches, dim=1)

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block that learns rotation and entanglement angles."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.rx_params = tq.ParameterVector("rx", self.n_wires)
        self.ry_params = tq.ParameterVector("ry", self.n_wires)
        self.rz_params = tq.ParameterVector("rz", self.n_wires)
        self.crx_params = tq.ParameterVector("crx", self.n_wires - 1)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.n_wires):
            self.rx_params[i](qdev, wires=i)
            self.ry_params[i](qdev, wires=i)
            self.rz_params[i](qdev, wires=i)
        for i in range(self.n_wires - 1):
            self.crx_params[i](qdev, wires=[i, i + 1])
        return tqf.measure_all(qdev, wires=list(range(self.n_wires)), measurement=tq.PauliZ)

class HybridNatModel(tq.QuantumModule):
    """
    Hybrid quantum model that merges quanvolution, quantum self‑attention and a linear classifier.
    """
    def __init__(self, n_wires: int = 4, num_classes: int = 10):
        super().__init__()
        self.n_wires = n_wires
        self.qfilter = QuantumQuanvolutionFilter()
        self.attention = QuantumSelfAttention(n_wires=self.n_wires)
        self.classifier = nn.Linear(4 * 14 * 14 + self.n_wires, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Quanvolution features
        quanv = self.qfilter(x)  # shape (bsz, 784)
        # Quantum self‑attention on a reduced representation
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
        attn_out = self.attention(qdev)  # (bsz, 4)
        # Concatenate quantum features
        features = torch.cat([quanv, attn_out], dim=1)
        logits = self.classifier(features)
        return logits

__all__ = ["HybridNatModel"]
