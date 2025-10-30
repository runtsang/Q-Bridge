import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionGen(nn.Module):
    """
    Enhanced quanvolution filter that uses a trainable quantum kernel
    to extract features from 2×2 patches.  The filter outputs both
    amplitude and phase, which are concatenated and fed into a
    hybrid classical‑quantum head.  The head can be swapped between
    a classical linear layer and a small quantum linear layer.
    """
    def __init__(self, in_channels: int = 1, out_classes: int = 10, use_quantum_head: bool = False):
        super().__init__()
        self.patch_size = 2
        self.n_wires = 4
        self.use_quantum_head = use_quantum_head

        # Encoder: map pixel values to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable quantum layer
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)), trainable=True)
        # Measurement: amplitude and phase
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Feature dimension: 8 values per 2×2 patch (amplitude & phase)
        self.features_dim = 8 * 14 * 14  # 14 patches per axis

        # Classical head
        self.classical_head = nn.Linear(self.features_dim, out_classes)

        # Quantum head
        if use_quantum_head:
            # Each class is represented by a qubit; the weight matrix
            # maps the full feature vector into rotation angles.
            self.q_head_weights = nn.Parameter(torch.randn(out_classes, self.features_dim))
            self.q_head_measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device

        # Ensure shape (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                # Gather 2×2 patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                # amplitude: absolute value, phase: sign
                amp = measurement.abs()
                phase = measurement.sign()
                patch_feat = torch.cat([amp, phase], dim=1)
                patches.append(patch_feat)

        features = torch.cat(patches, dim=1)  # shape (N, features_dim)

        if self.use_quantum_head:
            # Quantum linear head: each class is a qubit
            logits = torch.zeros(bsz, self.classical_head.out_features, device=device)
            for i in range(self.classical_head.out_features):
                qdev = tq.QuantumDevice(1, bsz=bsz, device=device)
                # Compute rotation angle as weighted sum of features
                angle = torch.matmul(features, self.q_head_weights[i])
                tq.RY(qdev, wires=0, params=angle)
                logits[:, i] = self.q_head_measure(qdev)
        else:
            logits = self.classical_head(features)

        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen"]
