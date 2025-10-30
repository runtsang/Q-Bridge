from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine

# Import the photonic‑style parameters and builder from the original QML seed
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program
# Import the quantum convolutional filter from Conv.py
from.Conv import Conv


class FraudDetectionQuantumModel:
    """
    Hybrid quantum‑classical fraud‑detection model.

    The model first runs a quantum convolutional filter to extract a
    scalar feature from a 2×2 patch.  This feature is then used to set
    the displacement parameters of a Strawberry Fields photonic program,
    which is executed on a Fock‑space simulator.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        conv_kernel_size: int = 2,
        conv_threshold: float = 127,
        shots: int = 100,
    ) -> None:
        # Quantum convolutional filter
        self.conv = Conv(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.fraud_params = fraud_params
        self.shots = shots
        # Fock‑space backend for the photonic circuit
        self.backend = Engine("fock", backend_options={"cutoff_dim": 5})

    def run(self, patch: np.ndarray) -> float:
        """
        Parameters
        ----------
        patch : np.ndarray
            2×2 array of pixel intensities.

        Returns
        -------
        float
            Fraud probability estimated by the photonic program.
        """
        # Quantum convolution → scalar feature
        conv_val = self.conv.run(patch)

        # Build a photonic program with displacement set to the quantum feature
        input_params = FraudLayerParameters(
            bs_theta=self.fraud_params.bs_theta,
            bs_phi=self.fraud_params.bs_phi,
            phases=self.fraud_params.phases,
            squeeze_r=self.fraud_params.squeeze_r,
            squeeze_phi=self.fraud_params.squeeze_phi,
            displacement_r=(conv_val, conv_val),
            displacement_phi=self.fraud_params.displacement_phi,
            kerr=self.fraud_params.kerr,
        )
        program = build_fraud_detection_program(input_params, [])

        # Execute the photonic program
        results = self.backend.run(program)
        # Return expectation value of mode 0 as the fraud score
        return results.expectation_value(0)


__all__ = ["FraudDetectionQuantumModel"]
