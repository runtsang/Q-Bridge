import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquantum import QuantumModule, QuantumDevice, GeneralEncoder, RandomLayer, MeasureAll, PauliZ

class ClassicalConvBackbone(nn.Module):
    """Standard 2‑D convolutional feature extractor."""
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.features(x)

class QuantumPatchEncoder(QuantumModule):
    """Two‑qubit quantum kernel applied to each 2×2 image patch."""
    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        device = QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(device, patch)
                self.random_layer(device)
                measurement = self.measure(device)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)   # shape: (bsz, 4*14*14)

class QuantumMixer(QuantumModule):
    """Quantum mixer that entangles the fused feature vector."""
    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = RandomLayer(n_ops=10, wires=list(range(n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, params):
        bsz = params.shape[0]
        device = QuantumDevice(self.n_wires, bsz=bsz, device=params.device)
        self.encoder(device, params)
        self.random_layer(device)
        return self.measure(device)

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical‑quantum model combining convolution, patch‑encoding, and a quantum mixer."""
    def __init__(self, in_channels=1, num_classes=10, n_qubits=4):
        super().__init__()
        self.classical_backbone = ClassicalConvBackbone(in_channels, out_channels=16)
        self.patch_encoder = QuantumPatchEncoder(n_wires=n_qubits)
        self.fusion = nn.Linear(16*7*7 + n_qubits*14*14, 128)
        self.param_linear = nn.Linear(128, n_qubits)
        self.quantum_mixer = QuantumMixer(n_wires=n_qubits)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        class_feat = self.classical_backbone(x).view(x.size(0), -1)
        patch_feat = self.patch_encoder(x)
        fused = torch.cat([class_feat, patch_feat], dim=1)
        fused = self.fusion(fused)
        q_params = self.param_linear(fused)
        mixer_out = self.quantum_mixer(q_params)
        logits = self.classifier(mixer_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
