import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Hybrid quanvolution filter combining a parameterised quantum kernel with a classical shortcut."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.n_wires = 4
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)
        self.shortcut_weight = nn.Parameter(torch.tensor(1.0))
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.param_circuit = tq.ParameterizedCircuit(
            [
                {"func": "ry", "wires": [0]},
                {"func": "ry", "wires": [1]},
                {"func": "ry", "wires": [2]},
                {"func": "ry", "wires": [3]},
                {"func": "cnot", "control": 0, "target": 1},
                {"func": "cnot", "control": 2, "target": 3},
                {"func": "ry", "wires": [0]},
                {"func": "ry", "wires": [1]},
                {"func": "ry", "wires": [2]},
                {"func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)
        shortcut = shortcut.view(bsz, -1)
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
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.param_circuit(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        qfeat = torch.cat(patches, dim=1)
        out = self.shortcut_weight * shortcut + qfeat
        return out

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the quanvolution filter followed by a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.log_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features) * self.log_scale
        logits = self.dropout(logits)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
