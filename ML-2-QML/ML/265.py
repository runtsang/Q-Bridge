import torch
from torch import nn
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

class FraudDetectionModel(nn.Module):
    """
    Classical neural network that mimics a layered photonic circuit.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.input_params = input_params
        self.layers_params = list(layers)

        modules = [self._layer_from_params(input_params, clip=False)]
        modules.extend(self._layer_from_params(lp, clip=True) for lp in self.layers_params)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules).to(self.device)

    def _layer_from_params(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return Layer().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classical model.
        """
        return self.model(x.to(self.device))

    def to_quantum(self):
        """
        Construct a PennyLane variational circuit that reproduces the same
        parameter set. Returns a callable that accepts a device and returns
        a QNode.
        """
        import pennylane as qml
        from pennylane import numpy as np

        def circuit(x, dev):
            @qml.qnode(dev)
            def qnode(inputs):
                # Encode inputs as displacements
                for i in range(2):
                    qml.Displacement(inputs[i], 0.0, wires=i)
                # Apply layers
                def apply_layer(params, clip):
                    # Beamsplitter
                    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
                    for i, phase in enumerate(params.phases):
                        qml.Rgate(phase, wires=i)
                    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                        qml.Sgate(r if not clip else _clip(r, 5.0), phi, wires=i)
                    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
                    for i, phase in enumerate(params.phases):
                        qml.Rgate(phase, wires=i)
                    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                        qml.Dgate(r if not clip else _clip(r, 5.0), phi, wires=i)
                    for i, k in enumerate(params.kerr):
                        qml.KerrGate(k if not clip else _clip(k, 1.0), wires=i)
                apply_layer(self.input_params, clip=False)
                for lp in self.layers_params:
                    apply_layer(lp, clip=True)
                # Measure expectation of photon number on mode 0
                return qml.expval(qml.NumberOperator(0))
            return qnode(x)
        return circuit

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
