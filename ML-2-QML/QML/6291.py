import pennylane as qml
import torch
from dataclasses import dataclass
from typing import Iterable, Tuple

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

def _apply_layer(params: FraudLayerParameters, clip: bool) -> None:
    qml.RY(params.bs_theta, wires=0)
    qml.RY(params.bs_phi, wires=1)
    for i, phase in enumerate(params.phases):
        qml.PhaseShift(phase, wires=i)
    for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5.0), wires=i)
    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0), wires=i)
    for i, k in enumerate(params.kerr):
        qml.CNOT(wires=[i, 1-i])
        qml.RZ(_clip(k, 1.0), wires=i)

class FraudDetectionHybrid:
    """
    Variational quantum circuit that mirrors the classical fraud‑detection
    architecture.  It accepts a 2‑dimensional input, encodes it into two qubits,
    and outputs a logit vector for the fraud classes via expectation values.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 num_classes: int = 2,
                 device: str = "default.qubit",
                 wires: Tuple[int, int] = (0, 1)) -> None:
        self.input_params = input_params
        self.layer_params = list(layer_params)
        self.num_classes = num_classes
        self.device = device
        self.wires = wires
        self._build_qnode()

    def _build_qnode(self):
        dev = qml.device(self.device, wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            _apply_layer(self.input_params, clip=False)
            for params in self.layer_params:
                _apply_layer(params, clip=True)
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in self.wires])

        self.qnode = circuit

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        logits = [self.qnode(inputs[i]) for i in range(batch_size)]
        return torch.stack(logits)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
