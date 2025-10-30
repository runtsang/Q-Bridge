"""Hybrid fraud‑detection model combining classical layers with a quantum kernel.

The module implements:
* A `QuanvolutionKernel` that applies a random two‑qubit quantum circuit to 2×2 patches,
  inspired by the quanvolution example.
* A `FraudDetectionHybridModel` that embeds the quantum kernel and follows
  the photonic‑style architecture from the original fraud‑detection seed.
* A convenient factory `build_fraud_detection_program` that returns the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import QuantumModule, QuantumDevice, GeneralEncoder, RandomLayer, MeasureAll, PauliZ


class FraudLayerParameters:
    """Parameters for a single fully‑connected fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class QuanvolutionKernel(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, height, width).
            For fraud data we assume a 2‑D representation (e.g. a 28×28 matrix).
        Returns
        -------
        torch.Tensor
            Concatenated measurement results for all 2×2 patches.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=device)

        patches = []
        for r in range(0, x.shape[1], 2):
            for c in range(0, x.shape[2], 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class FraudDetectionHybridModel(nn.Module):
    """Classical neural network that embeds a quantum kernel and follows the photonic‑style architecture."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        quantum_depth: int = 1,
    ):
        super().__init__()
        self.quantum = QuanvolutionKernel()
        self.classical_layers = nn.ModuleList()

        # first layer mirrors input_params but without clipping
        self.classical_layers.append(self._build_layer(input_params, clip=False))

        # subsequent layers clip weights
        for p in layers:
            self.classical_layers.append(self._build_layer(p, clip=True))

        self.head = nn.Linear(2, 1)

    def _build_layer(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [
                [params.bs_theta, params.bs_phi],
                [params.squeeze_r[0], params.squeeze_r[1]],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)

        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)

        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantum feature map
        qfeat = self.quantum(x)
        # reshape to match 2‑dim input for first classical layer
        qfeat = qfeat.view(x.shape[0], 2)
        out = qfeat
        for layer in self.classical_layers:
            out = layer(out)
        out = self.head(out)
        return out


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> FraudDetectionHybridModel:
    """Convenience factory mirroring the original seed."""
    return FraudDetectionHybridModel(input_params, layers)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybridModel"]
