import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum patch filter using a parameterized variational circuit."""
    def __init__(self, n_wires: int = 4, depth: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Input encoding: rotate each qubit by the pixel value
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational layer with learnable parameters
        self.var_layer = tq.variational.VariationalLayer(
            n_params=depth * n_wires * 3,
            n_wires=n_wires,
            layer_type="ryrzcnot",
            n_layers=depth,
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
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (batch, 4*14*14)

class Quanvolution__gen571(nn.Module):
    """Quantum‑inspired network with a variational patch filter."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen571"]
