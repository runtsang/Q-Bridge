import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

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

class ScaleShift(nn.Module):
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.scale = scale
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

class HybridQuanvolutionFraudNet(nn.Module):
    def __init__(self, fraud_params: List[FraudLayerParameters], dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.fraud_params = fraud_params
        self.n_wires = 4
        self.dev = qml.device(dev_name, wires=self.n_wires)
        self.qnode = qml.QNode(self._quanvolution_circuit, self.dev, interface="torch")
        self.layers = nn.ModuleList()
        input_dim = 4 * 14 * 14
        for params in fraud_params:
            self.layers.append(self._build_fraud_layer(input_dim, params))
            input_dim = 2
        self.final = nn.Linear(2, 1)

    def _quanvolution_circuit(self, img: torch.Tensor) -> torch.Tensor:
        img = img.reshape(28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[r:r+2, c:c+2].flatten()
                qml.RY(patch[0], wires=0)
                qml.RY(patch[1], wires=1)
                qml.RY(patch[2], wires=2)
                qml.RY(patch[3], wires=3)
                for _ in range(8):
                    qml.CNOT(wires=[0, 1])
                    qml.RX(torch.randn(1).item(), wires=0)
                    qml.CNOT(wires=[1, 2])
                    qml.RY(torch.randn(1).item(), wires=1)
                meas = torch.stack([qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)])
                patches.append(meas)
        return torch.cat(patches)

    def _build_fraud_layer(self, input_dim: int, params: FraudLayerParameters) -> nn.Module:
        weight = torch.tensor([[params.bs_theta]*input_dim, [params.bs_phi]*input_dim], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(input_dim, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        return nn.Sequential(linear, nn.Tanh(), ScaleShift(scale, shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            img = x[i, 0].flatten()
            q_out = self.qnode(img)
            outputs.append(q_out)
        q_outs = torch.stack(outputs)
        out = q_outs
        for layer in self.layers:
            out = layer(out)
        logits = self.final(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["FraudLayerParameters", "HybridQuanvolutionFraudNet"]
