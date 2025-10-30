import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum quanvolution filter that applies a parameterized two‑qubit ansatz
    to each 2×2 patch of the input image and returns the expectation
    values of Pauli‑Z on each qubit.
    """

    def __init__(self, n_wires=4, n_layers=4):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
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
                for layer in range(self.n_layers):
                    for w in range(self.n_wires):
                        qdev.ry(self.params[layer, w], wires=[w])
                    qdev.cx(0, 1)
                    qdev.cx(2, 3)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid quantum‑classical classifier: quantum quanvolution filter
    followed by a linear head.
    """

    def __init__(self, n_wires=4, num_classes=10):
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_wires=n_wires)
        self.linear = nn.Linear(n_wires * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
