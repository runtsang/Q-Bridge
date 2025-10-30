import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(nn.Module):
    """
    Quantum quanvolution filter using a parameterised circuit followed by a linear head.
    The circuit encodes 2×2 image patches into a 4‑qubit state and applies a variational layer.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2,
                 num_classes: int = 10):
        super().__init__()
        self.n_wires = 4
        # Encoder: map pixel intensities to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational layer: parameterised two‑qubit rotations
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear classification head
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 1, 28, 28)
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
        out = torch.cat(patches, dim=1)
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)

    def quantum_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        """
        Compute the fidelity between two quantum states represented as density matrices.
        """
        # Assume state1, state2 shape: (batch, 2**n)
        sqrt1 = torch.sqrt(state1)
        prod = sqrt1 * state2
        return torch.sum(prod, dim=1).pow(2)

__all__ = ["QuanvolutionHybrid"]
