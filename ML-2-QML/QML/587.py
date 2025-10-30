import torch
from torch import nn
import pennylane as qml
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

class FraudDetector(nn.Module):
    """
    Hybrid quantum-classical fraud detection model.
    Uses a PennyLane variational photonic circuit (continuous‑variable)
    followed by a classical linear classifier.
    """

    def __init__(
        self,
        layers: Iterable[FraudLayerParameters],
        device: str = "strawberryfields.fock",
        cutoff_dim: int = 10,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.dev = qml.device(device, wires=2, cutoff_dim=cutoff_dim)

        # Flatten parameters into a list of nn.ParameterDicts for trainability
        self.params = nn.ParameterList()
        for layer in self.layers:
            param_dict = nn.ParameterDict(
                {
                    "bs_theta": nn.Parameter(torch.tensor(layer.bs_theta)),
                    "bs_phi": nn.Parameter(torch.tensor(layer.bs_phi)),
                    "phases_0": nn.Parameter(torch.tensor(layer.phases[0])),
                    "phases_1": nn.Parameter(torch.tensor(layer.phases[1])),
                    "squeeze_r_0": nn.Parameter(torch.tensor(layer.squeeze_r[0])),
                    "squeeze_r_1": nn.Parameter(torch.tensor(layer.squeeze_r[1])),
                    "squeeze_phi_0": nn.Parameter(torch.tensor(layer.squeeze_phi[0])),
                    "squeeze_phi_1": nn.Parameter(torch.tensor(layer.squeeze_phi[1])),
                    "displacement_r_0": nn.Parameter(torch.tensor(layer.displacement_r[0])),
                    "displacement_r_1": nn.Parameter(torch.tensor(layer.displacement_r[1])),
                    "displacement_phi_0": nn.Parameter(torch.tensor(layer.displacement_phi[0])),
                    "displacement_phi_1": nn.Parameter(torch.tensor(layer.displacement_phi[1])),
                    "kerr_0": nn.Parameter(torch.tensor(layer.kerr[0])),
                    "kerr_1": nn.Parameter(torch.tensor(layer.kerr[1])),
                }
            )
            self.params.append(param_dict)

        # Classical classifier that maps quantum expectation values to output
        self.classifier = nn.Linear(2, num_classes)

        # Compile the variational circuit
        self._qnode = qml.qnode(
            self.dev,
            interface="torch",
            diff_method="backprop",
        )(self._circuit)

    def _circuit(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Variational continuous‑variable circuit.
        `inputs` should be a 1‑D tensor of length 2 (feature values).
        """
        # Prepare input state via displacement gates
        for i, val in enumerate(inputs):
            qml.Dgate(val, 0.0, wires=i)

        # Apply each layer of trainable gates
        for param_dict in self.params:
            # Beamsplitter
            qml.BSgate(
                param_dict["bs_theta"], param_dict["bs_phi"], wires=[0, 1]
            )
            # Phase shifters
            qml.Rgate(param_dict["phases_0"], wires=0)
            qml.Rgate(param_dict["phases_1"], wires=1)
            # Squeezing
            qml.Sgate(
                param_dict["squeeze_r_0"], param_dict["squeeze_phi_0"], wires=0
            )
            qml.Sgate(
                param_dict["squeeze_r_1"], param_dict["squeeze_phi_1"], wires=1
            )
            # Displacement
            qml.Dgate(
                param_dict["displacement_r_0"],
                param_dict["displacement_phi_0"],
                wires=0,
            )
            qml.Dgate(
                param_dict["displacement_r_1"],
                param_dict["displacement_phi_1"],
                wires=1,
            )
            # Kerr non‑linearity (if supported)
            try:
                qml.Kgate(param_dict["kerr_0"], wires=0)
                qml.Kgate(param_dict["kerr_1"], wires=1)
            except AttributeError:
                # Skip if the device does not provide Kgate
                pass

        # Measure expectation values of the number operator on each mode
        return torch.stack(
            [
                qml.expval(qml.NumberOperator(w))
                for w in range(2)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: run the variational circuit for each example
        and classify with the linear head.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            quantum_out = self._qnode(inputs[i])
            pred = self.classifier(quantum_out)
            outputs.append(pred)
        return torch.cat(outputs, dim=0)

__all__ = ["FraudLayerParameters", "FraudDetector"]
