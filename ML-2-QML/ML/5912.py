"""
FraudDetectionHybrid: A unified classical/quantum fraud detection model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple, Union

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single layer of the fraud model.
    The same structure is used for both the classical and quantum variants.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudDetectionHybrid:
    """
    Unified fraud detection model that can be instantiated either as a
    classical PyTorch network or as a hybrid quantum‑classical model using
    Strawberry‑Fields.  The class exposes a common API for building,
    training and evaluating the model.
    """

    def __init__(
        self,
        mode: str,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        mode : str
            Either ``"classical"`` or ``"quantum"``.
        input_params : FraudLayerParameters
            Parameters for the first layer.
        layers : Iterable[FraudLayerParameters]
            Parameters for the hidden layers.
        device : str, optional
            Torch device to use.
        """
        assert mode in ("classical", "quantum")
        self.mode = mode
        self.device = device
        self.input_params = input_params
        self.layers = list(layers)

        if mode == "classical":
            self.model = self._build_classical()
            self.head = None
        else:
            self.model = None
            # Small classical head that maps the 2‑dim state to a scalar
            self.head = nn.Linear(2, 1, bias=False).to(self.device)

    # ------------------------------------------------------------------
    # Classical model construction
    # ------------------------------------------------------------------
    def _build_classical(self) -> nn.Sequential:
        modules: List[nn.Module] = []

        def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
                    out = self.activation(self.linear(inputs))
                    out = out * self.scale + self.shift
                    return out

            return Layer()

        modules.append(_layer_from_params(self.input_params, clip=False))
        modules.extend(_layer_from_params(l, clip=True) for l in self.layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules).to(self.device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,
        quantum_circuit: Union["QuantumFraudCircuit", None] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the selected mode.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 2).
        quantum_circuit : Union[QuantumFraudCircuit, None], optional
            Quantum circuit object used in quantum mode.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1).
        """
        if self.mode == "classical":
            return self.model(inputs)
        else:
            assert quantum_circuit is not None, (
                "Quantum circuit must be supplied in quantum mode."
            )
            # Execute the circuit and obtain measurement results
            state_vec = quantum_circuit.execute(inputs)
            # Pass through the classical head
            return self.head(state_vec)

    # ------------------------------------------------------------------
    # Parameters for optimisation
    # ------------------------------------------------------------------
    def parameters(self):
        if self.mode == "classical":
            return self.model.parameters()
        else:
            return self.head.parameters()

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
