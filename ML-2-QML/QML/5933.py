import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumNAT(tq.QuantumModule):
    """Hybrid quantum‑classical model that applies a quantum kernel to classical convolutional patches."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Mapping 64‑dim patch to 4‑dim quantum feature
        self.pre_quantum = nn.Linear(64, 4)

        self.n_wires = 4
        # Encoder that maps 4‑dim feature to qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(36)      # 3x3 patches * 4 qubits each
        self.fc = nn.Sequential(
            nn.Linear(36, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)           # (bsz,16,7,7)
        patches = []
        for r in range(0, 7, 2):
            for c in range(0, 7, 2):
                patch = feat[:, :, r:r+2, c:c+2]      # (bsz,16,2,2)
                patch_flat = patch.view(bsz, -1)      # (bsz,64)
                qfeat = self.pre_quantum(patch_flat)  # (bsz,4)
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
                self.encoder(qdev, qfeat)
                self.q_layer(qdev)
                meas = self.measure(qdev)              # (bsz,4)
                patches.append(meas)
        quantum_features = torch.cat(patches, dim=1)  # (bsz,36)
        quantum_features = self.norm(quantum_features)
        out = self.fc(quantum_features)
        return out

__all__ = ["HybridQuantumNAT"]
