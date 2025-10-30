"""Quantum-enhanced quanvolutional model using a trainable variational circuit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionModel(tq.QuantumModule):
    """
    Quantum version of the quanvolution model.
    Each 2x2 patch is encoded with single‑qubit rotations, passed through a
    parameterised variational layer, and measured. The resulting 4‑dimensional
    feature vector per patch is flattened and fed into a classical linear head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        depth_multiplier: int = 1,
        n_variational_layers: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = 4  # one qubit per pixel in a 2x2 patch
        # Encoder: map pixel intensities to ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational layer with trainable parameters
        self.var_layer = tq.ParameterizedLayer(
            n_ops=n_variational_layers * 4,
            ops=[["ry", "rz", "cx"]],
            wires=list(range(self.n_wires)),
            seed=seed,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear classifier
        self.feature_dim = (28 // 2) * (28 // 2) * 4
        self.classifier = nn.Linear(self.feature_dim, num_classes)

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
        features = torch.cat(patches, dim=1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionModel"]
