"""Quantum hybrid filter with skip connection and state‑vector output.

The implementation follows the original Quanvolution filter but
adds a classical projection of the raw image to create a residual
branch.  The measurement results of a random two‑qubit circuit
are concatenated across all 2×2 patches, added to the classical
projection, and optionally passed through a linear head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    def __init__(self, num_classes: int = 10, use_classifier: bool = True):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical projection to match the quantum feature size
        self.classic_proj = nn.Linear(28 * 28, 4 * 14 * 14)
        self.relu = nn.ReLU(inplace=True)
        self.use_classifier = use_classifier
        if use_classifier:
            self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x_reshaped = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x_reshaped[:, r, c],
                        x_reshaped[:, r, c + 1],
                        x_reshaped[:, r + 1, c],
                        x_reshaped[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        quantum_out = torch.cat(patches, dim=1)
        classic_out = self.classic_proj(x.view(bsz, -1))
        out = quantum_out + classic_out
        out = self.relu(out)
        if self.use_classifier:
            logits = self.linear(out)
            return F.log_softmax(logits, dim=-1)
        else:
            return out

__all__ = ["QuanvolutionHybrid"]
