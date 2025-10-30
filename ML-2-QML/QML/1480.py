"""
FraudDetectionModel – Quantum implementation using Pennylane.

The quantum side reproduces the photonic layer semantics in a qubit‑based
variational circuit.  Each layer is a sequence of parameterised rotations
followed by a CNOT entangler.  The model outputs the expectation value
of Pauli‑Z on qubit 0, which serves as the fraud‑risk score.

Key features
------------
* ``FraudLayerParameters`` – stores the same parameters as the classical model.
* ``FraudDetectionModel`` – a PennyLane ``qml.QNode`` that accepts a 2‑dimensional
  input vector and returns a scalar prediction.
* Dropout and batch‑norm analogues are omitted; the circuit is fully
  differentiable and compatible with PennyLane's autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single photonic layer, reused for the quantum model.
    """

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    wires: Sequence[int],
    params: FraudLayerParameters,
    clip: bool,
    q: qml.QubitDevice,
) -> None:
    """
    Map a classical layer to a sequence of qubit gates.

    Parameters
    ----------
    wires : Sequence[int]
        Wire indices for the two qubits.
    params : FraudLayerParameters
        Layer parameters.
    clip : bool
        Whether to clip values to a safe range.
    q : qml.QubitDevice
        The device to which gates will be applied.
    """
    theta = params.bs_theta
    phi = params.bs_phi
    phase0, phase1 = params.phases
    r0, r1 = params.squeeze_r
    phi0, phi1 = params.squeeze_phi
    dr0, dr1 = params.displacement_r
    dphi0, dphi1 = params.displacement_phi
    k0, k1 = params.kerr

    if clip:
        theta = _clip(theta, 5.0)
        phi = _clip(phi, 5.0)
        r0 = _clip(r0, 5.0)
        r1 = _clip(r1, 5.0)
        dr0 = _clip(dr0, 5.0)
        dr1 = _clip(dr1, 5.0)
        k0 = _clip(k0, 1.0)
        k1 = _clip(k1, 1.0)

    # First rotation block – emulate beam‑splitter angles
    qml.RZ(theta, wires=wires[0])
    qml.RZ(phi, wires=wires[1])

    # Phase shifts
    qml.RZ(phase0, wires=wires[0])
    qml.RZ(phase1, wires=wires[1])

    # Squeezing – mapped to RX rotations
    qml.RX(r0, wires=wires[0])
    qml.RX(r1, wires=wires[1])

    # Entanglement
    qml.CNOT(wires=wires)

    # Second rotation block – emulate second beam‑splitter
    qml.RZ(theta, wires=wires[0])
    qml.RZ(phi, wires=wires[1])

    # Phase shifts again
    qml.RZ(phase0, wires=wires[0])
    qml.RZ(phase1, wires=wires[1])

    # Displacement – mapped to RY rotations
    qml.RY(dr0, wires=wires[0])
    qml.RY(dr1, wires=wires[1])

    # Kerr – small additional rotations
    qml.RZ(k0, wires=wires[0])
    qml.RZ(k1, wires=wires[1])


# --------------------------------------------------------------------------- #
# Quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionModel:
    """
    Quantum fraud‑detection model implemented as a PennyLane QNode.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for any number of hidden layers.
    device : qml.Device, optional
        PennyLane device to run the circuit on.  Defaults to
        ``qml.device('default.qubit', wires=2)``.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: qml.devices.Device | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)

        self.device = device or qml.device("default.qubit", wires=2)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(x: np.ndarray) -> np.ndarray:
            # Encode classical input as initial rotations
            qml.RX(x[0], wires=0)
            qml.RX(x[1], wires=1)

            # Apply input layer
            _apply_layer([0, 1], self.input_params, clip=False, q=self.device)

            # Apply hidden layers
            for layer in self.layers:
                _apply_layer([0, 1], layer, clip=True, q=self.device)

            # Output: expectation of Pauli‑Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : array-like, shape (2,)
            2‑dimensional input vector.

        Returns
        -------
        torch.Tensor
            Scalar fraud‑risk score.
        """
        return self._circuit(np.array(x, dtype=np.float64))

    @staticmethod
    def from_config(
        config: dict,
        *,
        device: qml.devices.Device | None = None,
    ) -> "FraudDetectionModel":
        """
        Build a model from a configuration dictionary.

        The dictionary should contain ``input_params`` and ``layers`` keys,
        each mapping to a list of parameter dictionaries matching
        ``FraudLayerParameters``.
        """
        input_params = FraudLayerParameters(**config["input_params"])
        layers = [FraudLayerParameters(**p) for p in config["layers"]]
        return FraudDetectionModel(input_params, layers, device=device)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
