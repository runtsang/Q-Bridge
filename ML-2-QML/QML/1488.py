import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class VariationalQuanvolutionFilter(tq.QuantumModule):
    """Parameterised quantum filter that learns a 2×2 kernel."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Data encoding layer – one RY per image pixel
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational layer with learnable rotation angles and CNOTs
        self.var_layer = tq.ParameterizedLayer(
            n_ops=4 * n_layers,
            ops=[
                {"func": "ry", "wires": [0]},
                {"func": "ry", "wires": [1]},
                {"func": "ry", "wires": [2]},
                {"func": "ry", "wires": [3]},
            ],
            n_layers=n_layers,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
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

class QuanvolutionClassifier(nn.Module):
    """Hybrid quantum–classical classifier."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = VariationalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["VariationalQuanvolutionFilter", "QuanvolutionClassifier"]
