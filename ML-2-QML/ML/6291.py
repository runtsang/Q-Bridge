import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_classical_model(input_params: FraudLayerParameters,
                         layers: Iterable[FraudLayerParameters],
                         num_classes: int = 2) -> nn.Sequential:
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, num_classes))
    return nn.Sequential(*modules)

class FraudDetectionHybrid:
    """
    Hybrid fraud‑detection model that trains a classical feed‑forward network
    and a variational quantum circuit in parallel.  The two sub‑models are
    regularised to produce similar logits, which helps stabilise training
    when the quantum back‑end is noisy or limited in depth.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 num_classes: int = 2,
                 device: str = "cpu",
                 reg_weight: float = 0.1,
                 learning_rate: float = 1e-3) -> None:
        self.device = torch.device(device)
        self.classical = build_classical_model(input_params,
                                               layer_params,
                                               num_classes).to(self.device)
        self.quantum = None
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.Adam(self.classical.parameters(),
                                          lr=learning_rate)

    def set_quantum(self, quantum_callable):
        """
        Attach a quantum callable that takes a batch of 2‑dimensional
        inputs and returns a torch.Tensor of shape (batch, num_classes).
        """
        self.quantum = quantum_callable

    def _consistency_loss(self, cls_logits: torch.Tensor,
                          q_logits: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(cls_logits, q_logits)

    def train_step(self,
                   inputs: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
        self.classical.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        cls_logits = self.classical(inputs)
        if self.quantum is None:
            raise RuntimeError("Quantum callable not set.")
        q_logits = self.quantum(inputs)
        loss_cls = F.cross_entropy(cls_logits, labels)
        loss_cons = self._consistency_loss(cls_logits, q_logits)
        loss = loss_cls + self.reg_weight * loss_cons
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        self.classical.eval()
        with torch.no_grad():
            return self.classical(inputs.to(self.device))

__all__ = ["FraudLayerParameters", "build_classical_model", "FraudDetectionHybrid"]
