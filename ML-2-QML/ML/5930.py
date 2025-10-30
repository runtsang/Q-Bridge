import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable, Tuple

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical patch‑wise filter that emulates a 2×2 quantum kernel."""
    def __init__(self, patch_size: int = 2, n_out: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.n_out = n_out
        self.patch_net = nn.Sequential(
            nn.Linear(patch_size * patch_size, 8),
            nn.ReLU(),
            nn.Linear(8, n_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(bsz, c, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.patch_size * self.patch_size)
        out = self.patch_net(patches)
        out = out.view(bsz, -1, self.n_out)
        out = out.permute(0, 2, 1).reshape(bsz, -1)
        return out

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Trainable quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_params: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.var_layer = tq.QuantumLayer()
        self.var_layer.add_params(n_params)
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
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class Quanvolution(nn.Module):
    """Hybrid classifier that can use either the classical or quantum filter."""
    def __init__(self, mode: str = "quantum", num_features: int = 4 * 14 * 14, num_classes: int = 10):
        super().__init__()
        if mode == "quantum":
            self.feature_extractor = QuantumQuanvolutionFilter()
        else:
            self.feature_extractor = ClassicalQuanvolutionFilter()
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

def build_classifier_network(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a classical network that mimics the structure of the quantum
    classifier circuit. Returns (network, encoding, weight_sizes, observables).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = [
    "ClassicalQuanvolutionFilter",
    "QuantumQuanvolutionFilter",
    "Quanvolution",
    "build_classifier_network",
]
