import pennylane as qml
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

class FraudDetectionModel:
    """
    Variational photonic circuit that mirrors the classical fraud detection network.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev: str | qml.Device = "default.qubit",
        wires: int = 2,
        shots: int = 5000,
    ) -> None:
        self.input_params = input_params
        self.layers_params = list(layers)
        self.dev = qml.device(dev, wires=wires, shots=shots)

        def circuit(inputs):
            # Encode inputs as displacements
            for i in range(wires):
                qml.Displacement(inputs[i], 0.0, wires=i)

            def apply_layer(params, clip):
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

            # Return expectation of photon number in mode 0
            return qml.expval(qml.NumberOperator(0))

        self.qnode = qml.QNode(circuit, self.dev)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum circuit on the given input vector.
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        result = self.qnode(inputs)
        return torch.tensor(result, dtype=torch.float32)

    def to_classical(self):
        """
        Construct a classical PyTorch model that reproduces the same parameter set.
        """
        import torch
        from torch import nn

        class ClassicalModel(nn.Module):
            def __init__(self, input_params, layers):
                super().__init__()
                self.input_params = input_params
                self.layers_params = layers
                modules = [self._layer_from_params(input_params, clip=False)]
                modules.extend(self._layer_from_params(lp, clip=True) for lp in layers)
                modules.append(nn.Linear(2, 1))
                self.model = nn.Sequential(*modules)

            def _layer_from_params(self, params, clip):
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
                        outputs = self.activation(self.linear(inputs))
                        outputs = outputs * self.scale + self.shift
                        return outputs

                return Layer()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)

        return ClassicalModel(self.input_params, self.layers_params)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
