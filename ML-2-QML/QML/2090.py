import pennylane as qml
import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class FraudLayerParameters:
    """
    Parameters for a variational quantum layer.

    Attributes
    ----------
    rot_angles : np.ndarray
        Shape (2, 3) array of rotation angles (Rx, Ry, Rz) for each qubit.
    cz : bool
        Whether to apply a CZ gate between the two qubits.
    clip : bool
        If True, clip angles to the interval [-π, π] to keep them within a
        stable region for the simulator.
    """
    rot_angles: np.ndarray
    cz: bool
    clip: bool = False


class FraudDetectionModel:
    """
    Quantum‑classical fraud‑detection model using a PennyLane variational circuit.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent hidden layers.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        self.device = qml.device("default.qubit", wires=2)
        self.params = [input_params] + list(layers)
        self.circuit = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the variational circuit.

        Parameters
        ----------
        x : torch.Tensor
            2‑element input vector.

        Returns
        -------
        torch.Tensor
            Expectation value of Pauli‑Z on qubit 0 as the model output.
        """
        # Encode the classical input
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Apply all layers sequentially
        for layer in self.params:
            self._apply_layer(layer)

        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, layer: FraudLayerParameters) -> None:
        angles = layer.rot_angles
        if layer.clip:
            angles = np.clip(angles, -np.pi, np.pi)

        # Rotations for each qubit
        qml.Rot(*angles[0], wires=0)
        qml.Rot(*angles[1], wires=1)

        if layer.cz:
            qml.CZ(wires=[0, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper to call the PennyLane QNode with a torch input.

        Parameters
        ----------
        x : torch.Tensor
            2‑element input vector.

        Returns
        -------
        torch.Tensor
            Model prediction (a scalar).
        """
        return self.circuit(x)
