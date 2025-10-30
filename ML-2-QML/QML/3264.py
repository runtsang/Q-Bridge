"""Hybrid quantum model: quanvolutional filter implemented with Pennylane and fraud‑detection style head."""

import pennylane as qml
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumFilter(nn.Module):
    """Quantum convolutional filter that processes 2×2 image patches."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.n_wires = 4
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        rng = torch.randn
        self.encoder_params = {
            "ry0": rng(1).item(),
            "ry1": rng(1).item(),
            "ry2": rng(1).item(),
            "ry3": rng(1).item(),
        }
        self.random_layer_params = [rng(1).item() for _ in range(8)]

    def _circuit(self, data, params):
        qml.RY(params["ry0"], wires=0)
        qml.RY(params["ry1"], wires=1)
        qml.RY(params["ry2"], wires=2)
        qml.RY(params["ry3"], wires=3)
        for theta in self.random_layer_params:
            qml.RX(theta, wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        patches = []
        x = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                data_np = data.cpu().numpy()
                out = []
                for i in range(bsz):
                    @qml.qnode(self.dev, interface="torch")
                    def circuit():
                        return self._circuit(data_np[i], self.encoder_params)

                    out.append(circuit())
                patch_out = torch.stack(out)
                patches.append(patch_out)
        return torch.cat(patches, dim=1)


class _FraudLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(
            torch.tensor(params.displacement_r, dtype=torch.float32)
        )
        self.shift = nn.Parameter(
            torch.tensor(params.displacement_phi, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_FraudLayer(input_params, clip=False)]
    modules += [_FraudLayer(l, clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridQuanvolution(nn.Module):
    """Quantum hybrid model: quanvolutional filter followed by fraud‑detection style head."""

    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        head_layers: int = 3,
    ):
        super().__init__()
        self.filter = QuantumFilter()
        rng = torch.randn
        input_params = FraudLayerParameters(
            bs_theta=rng(1).item(),
            bs_phi=rng(1).item(),
            phases=(rng(1).item(), rng(1).item()),
            squeeze_r=(rng(1).item(), rng(1).item()),
            squeeze_phi=(rng(1).item(), rng(1).item()),
            displacement_r=(rng(1).item(), rng(1).item()),
            displacement_phi=(rng(1).item(), rng(1).item()),
            kerr=(rng(1).item(), rng(1).item()),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=rng(1).item(),
                bs_phi=rng(1).item(),
                phases=(rng(1).item(), rng(1).item()),
                squeeze_r=(rng(1).item(), rng(1).item()),
                squeeze_phi=(rng(1).item(), rng(1).item()),
                displacement_r=(rng(1).item(), rng(1).item()),
                displacement_phi=(rng(1).item(), rng(1).item()),
                kerr=(rng(1).item(), rng(1).item()),
            )
            for _ in range(head_layers - 1)
        ]
        self.head = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        flat = features.view(features.size(0), -1)
        logits = self.head(flat)
        return torch.log(logits, dim=-1)


__all__ = ["HybridQuanvolution", "FraudLayerParameters", "build_fraud_detection_program"]
