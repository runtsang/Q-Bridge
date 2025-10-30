import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilterQ(tq.QuantumModule):
    """Quantum quanvolution filter using 2×2 image patches."""
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        # Encode each pixel of a 2×2 patch with an Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
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
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

class HybridConvNet(tq.QuantumModule):
    """
    Quantum‑augmented convolutional network.

    Mirrors the classical HybridConvNet but replaces the
    quanvolution filter with a true variational circuit
    (GeneralEncoder → RandomLayer → trainable RX/RY).
    The head is a standard linear layer that maps the
    measured Pauli‑Z expectation values to logits.
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 1,
        num_wires: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        # Classical depthwise conv to reduce dimensionality
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=2, stride=2)
        # Quantum filter
        self.quanv = QuanvolutionFilterQ(num_wires=num_wires)
        # Compute head dimension using a dummy input
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat = self.forward_features(dummy)
        self.head = nn.Linear(feat.shape[1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.quanv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)
