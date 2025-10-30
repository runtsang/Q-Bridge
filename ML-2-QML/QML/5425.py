import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
#  Classical CNN backbone (shared with the ML version)
# --------------------------------------------------------------------------- #
class _CNNBackbone(nn.Module):
    """2‑layer ConvNet mirroring the original QFCModel architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# --------------------------------------------------------------------------- #
#  Quantum fully‑connected layer (Q‑FC)
# --------------------------------------------------------------------------- #
class QFCLayer(tq.QuantumModule):
    """Encodes a classical vector into qubits, applies a random circuit, and measures."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=20, wires=range(n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
#  Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """Variational attention that uses a small quantum circuit to produce weighted features."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=10, wires=range(n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
#  Hybrid quantum‑classical model
# --------------------------------------------------------------------------- #
class QuantumNAT__gen318(tq.QuantumModule):
    """Hybrid CNN‑Quantum‑Attention model for image classification."""
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _CNNBackbone()
        self.pre_fc   = nn.Linear(16 * 7 * 7, 64)      # Classical reduction
        self.fc_layer = QFCLayer(n_wires=64)           # Quantum FC
        self.attn_layer = QuantumSelfAttention(n_wires=64)  # Quantum attention
        self.out = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        flattened = features.view(x.size(0), -1)
        pre_fc_out = self.pre_fc(flattened)
        fc_out = self.fc_layer(pre_fc_out)
        attn_out = self.attn_layer(fc_out)
        logits = self.out(attn_out)
        return self.norm(logits)

__all__ = ["QuantumNAT__gen318"]
