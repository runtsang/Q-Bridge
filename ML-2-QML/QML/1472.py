"""Hybrid quanvolution with a learnable variational layer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionPlus(tq.QuantumModule):
    """Hybrid quanvolution filter with a ParameterizedLayer and a linear head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.n_wires = 4

        # Encode pixel values into qubit states via Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Learnable variational layer
        self.var_layer = self._build_var_layer()

        # Measurement of all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear head
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def _build_var_layer(self) -> tq.ParameterizedLayer:
        """Construct a small parameterized circuit with entanglement."""
        def circuit(qdev: tq.QuantumDevice, params: torch.Tensor):
            # First set of rotations and entanglement
            tq.ry(qdev, 0, params[0])
            tq.ry(qdev, 1, params[1])
            tq.cnot(qdev, 0, 1)
            tq.ry(qdev, 2, params[2])
            tq.ry(qdev, 3, params[3])
            tq.cnot(qdev, 2, 3)

            # Second set
            tq.ry(qdev, 0, params[4])
            tq.ry(qdev, 1, params[5])
            tq.cnot(qdev, 0, 1)
            tq.ry(qdev, 2, params[6])
            tq.ry(qdev, 3, params[7])
            tq.cnot(qdev, 2, 3)

        return tq.ParameterizedLayer(
            circuit=circuit,
            n_params=8,
            wires=list(range(self.n_wires)),
            trainable=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Prepare a patch of 4 pixels
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1
                )

                # Quantum device for this patch
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

                # Encode pixel intensities
                self.encoder(qdev, data)

                # Variational layer
                self.var_layer(qdev)

                # Measurement
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        # Concatenate all patch measurements
        features = torch.cat(patches, dim=1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionPlus"]
