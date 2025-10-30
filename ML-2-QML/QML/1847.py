"""Quantum variational quanvolution filter with a classical linear head.

This module implements the quanvolution idea using a learnable
parameterized quantum kernel instead of a fixed random circuit.
The encoder maps each 2×2 image patch to a 4‑qubit state, the
ansatz is a 8‑operation trainable circuit, and Pauli‑Z measurements
produce 4 real features per patch.  The flattened features are
fed into a linear classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Variational quanvolution filter + linear classifier.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    in_channels : int
        Number of input channels (default 1 for MNIST).
    out_channels : int
        Number of output channels per 2×2 patch (default 4).
    kernel_size : int
        Size of the convolution kernel (default 2).
    stride : int
        Stride of the convolution (default 2).
    n_wires : int
        Number of qubits used for each patch (default 4).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoding: 4 separate Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational ansatz: 8 parametrized operations
        self.q_layer = tq.ParameterizedLayer(
            n_ops=8, wires=list(range(self.n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        device = x.device
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
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
