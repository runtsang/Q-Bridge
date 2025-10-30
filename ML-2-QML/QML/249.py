"""Hybrid quanvolution model using torchquantum.

The class ``QuanvolutionNet`` implements a 2×2 patch‑wise quantum
kernel followed by a linear classifier.  It mirrors the classical
``QuanvolutionNet`` API so that the two modules are interchangeable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionNet(nn.Module):
    """
    Quantum 2×2 patch encoder with a trainable variational circuit.
    The circuit is parameterised by ``self.q_layer`` and can be
    optimised jointly with the linear head.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 num_classes: int = 10, kernel_size: int = 2,
                 stride: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: map pixel intensities to qubit rotations.
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        # Trainable variational layer with 8 two‑qubit gates.
        self.q_layer = tq.ParameterizedLayer(
            n_ops=8, n_wires=self.n_wires, params_per_op=1
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def _patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode all 2×2 patches of the input image using the quantum
        circuit and return a batch‑wise feature vector.
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, 1, 28, 28)``.
        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape ``(batch, num_classes)``.
        """
        features = self._patch_features(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
