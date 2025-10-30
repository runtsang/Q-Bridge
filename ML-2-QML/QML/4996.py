import torch
import torch.nn as nn
import torchquantum as tq

class QuantumSelfAttention(tq.QuantumModule):
    """Lightweight quantum self‑attention block implemented with random layers."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=15, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev)

class QuantumFeatureExtractor(tq.QuantumModule):
    """Encodes a 4‑dimensional patch into a quantum state and returns measurement amplitudes."""
    def __init__(self, n_wires: int = 4):
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
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑centric counterpart of QuanvolutionHybrid.
    Uses a quantum patch filter, a quantum feature extractor,
    a quantum self‑attention block, and a classical linear head.
    """
    def __init__(self, n_wires: int = 4, num_classes: int = 10):
        super().__init__()
        self.n_wires = n_wires
        self.qfilter = QuantumFeatureExtractor(n_wires)
        self.qattention = QuantumSelfAttention(n_wires)
        self.linear = nn.Linear(n_wires * 3, num_classes)  # 3 * n_wires: filter + attention + raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # Extract 2×2 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r : r + 2, c : c + 2].reshape(bsz, -1)
                patches.append(patch)
        patches = torch.cat(patches, dim=1)  # (bs, 4 * 14 * 14)

        # Quantum filter features
        filt_feats = self.qfilter(qdev, patches)  # (bs, n_wires)

        # Quantum self‑attention
        attn_feats = self.qattention(qdev)  # (bs, n_wires)

        # Raw classical flatten (for additional context)
        raw_feats = patches.view(bsz, -1)[:, :self.n_wires]  # take first n_wires dims

        combined = torch.cat([filt_feats, attn_feats, raw_feats], dim=1)
        logits = self.linear(combined)
        return torch.nn.functional.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
