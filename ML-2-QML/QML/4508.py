"""Quantum counterpart of the hybrid autoencoder."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, Estimator as EstimatorBackend
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN


class AutoencoderGen192Q:
    """Hybrid quantum autoencoder that mirrors the classical AutoencoderGen192."""

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        num_qubits: int = 10,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.device = device

        # Feature‑map circuit for the variational autoencoder
        self.feature_map = RealAmplitudes(self.num_qubits, reps=3)
        self.input_params = list(self.feature_map.parameters)

        # Sampler for state‑vector probabilities
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.feature_map,
            input_params=self.input_params,
            weight_params=self.input_params,
            interpret=lambda x: x,
            output_shape=(self.num_qubits,),
            sampler=self.sampler,
        )

        # Simple regression circuit: one qubit with Ry(x)
        x = Parameter("x")
        reg_circuit = QuantumCircuit(1)
        reg_circuit.ry(x, 0)
        reg_observable = SparsePauliOp.from_list([("Z", 1)])
        self.reg_estimator = EstimatorQNN(
            circuit=reg_circuit,
            observables=reg_observable,
            input_params=[x],
            weight_params=[],
            estimator=EstimatorBackend(),
        )

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass returning a reconstruction and a regression output.

        Parameters
        ----------
        inputs : np.ndarray
            Input images of shape (batch, C, H, W) with values in [0, 1].

        Returns
        -------
        reconstruction : np.ndarray
            Reconstructed images of shape (batch, C, H, W).
        regression : np.ndarray
            Regression predictions of shape (batch, 1).
        """
        batch = inputs.shape[0]
        # Flatten images and normalise to [0, 1]
        flat_inputs = inputs.reshape(batch, -1)
        # Ensure the number of features matches the circuit qubits
        if flat_inputs.shape[1] > self.num_qubits:
            flat_inputs = flat_inputs[:, : self.num_qubits]
        elif flat_inputs.shape[1] < self.num_qubits:
            pad = np.zeros((batch, self.num_qubits - flat_inputs.shape[1]))
            flat_inputs = np.concatenate([flat_inputs, pad], axis=1)

        # Quantum autoencoder: obtain probability vector
        probs = self.qnn.forward(flat_inputs)
        # Map probabilities to image shape
        recon_flat = probs.reshape(batch, self.input_dim, 28, 28)
        reconstruction = recon_flat

        # Regression: use first probability as input to the regression circuit
        reg_input = probs[:, 0]
        reg_output = self.reg_estimator.forward(x=reg_input).output
        regression = reg_output.reshape(batch, 1)

        return reconstruction, regression


__all__ = ["AutoencoderGen192Q"]
