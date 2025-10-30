"""
Quantum‑kernel quanvolution filter with a parameter‑shared variational ansatz
and a classical linear classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Parameter‑shared variational quanvolution filter.
    """
    def __init__(self,
                 n_wires: int = 4,
                 n_layers: int = 3,
                 n_params_per_layer: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.n_params_per_layer = n_params_per_layer
        # Encoder: Ry on each wire
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        # Variational parameters (shared across all patches)
        self.variational_params = nn.Parameter(
            torch.randn(n_layers, n_params_per_layer)
        )
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Concatenated measurement results from all 2x2 patches.
        """
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
                # Apply variational layers with shared parameters
                for layer_idx in range(self.n_layers):
                    params = self.variational_params[layer_idx]
                    for i, wire in enumerate(range(self.n_wires)):
                        # Ry and Rz gates with layer parameters
                        qdev.ry(params[i % self.n_params_per_layer], wire)
                        qdev.rz(params[(i + 1) % self.n_params_per_layer], wire)
                    # Entanglement: CNOT between adjacent wires in a ring
                    for i in range(self.n_wires):
                        qdev.cx(i, (i + 1) % self.n_wires)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network using the QuanvolutionHybrid filter followed by a linear head.
    """
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionHybrid()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid", "QuanvolutionClassifier"]
