"""Hybrid estimator autoencoder quantum implementation.

This module builds a variational quantum circuit that mimics the
classical autoencoder architecture: an input encoding layer,
a RealAmplitudes ansatz for the latent space, a domain‑wall
swap test that acts as a measurement of similarity, and a
single‑bit measurement interpreted as a scalar output.  The
circuit is fed to a `SamplerQNN` which provides a differentiable
forward pass suitable for gradient‑based optimisation.

The class name is identical to the classical version so that
downstream code can treat either backend uniformly.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridEstimatorAutoencoder:
    """Quantum hybrid autoencoder with a single‑bit regression output."""
    def __init__(
        self,
        num_features: int,
        num_latent: int = 3,
        num_trash: int = 2,
        ansatz_reps: int = 5,
    ) -> None:
        # Build the core circuit
        self.qc = self._build_circuit(num_features, num_latent, num_trash, ansatz_reps)

        # Sampler backend
        self.sampler = Sampler()

        # Wrap in a SamplerQNN for differentiable execution
        self.qnn = SamplerQNN(
            circuit=self.qc,
            input_params=self._input_params(num_features),
            weight_params=self._weight_params(num_latent, num_trash),
            interpret=lambda x: float(x[0]),  # map measurement to scalar
            output_shape=1,
            sampler=self.sampler,
        )

    def _input_params(self, num_features: int) -> list[ParameterVector]:
        """Return the ParameterVector used for feature encoding."""
        return [ParameterVector(f"input_{i}" for i in range(num_features))]

    def _weight_params(self, num_latent: int, num_trash: int) -> list[ParameterVector]:
        """Return the ParameterVectors for the variational ansatz."""
        return [ParameterVector(f"weight_{i}" for i in range(num_latent + num_trash))]

    def _build_circuit(
        self,
        num_features: int,
        num_latent: int,
        num_trash: int,
        reps: int,
    ) -> QuantumCircuit:
        """Constructs a circuit that encodes inputs, runs a RealAmplitudes
        ansatz for the latent space, applies a domain‑wall swap test,
        and measures a single auxiliary qubit."""
        qr = QuantumRegister(num_features + num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature encoding
        fe = RawFeatureVector(num_features)
        qc.compose(fe, range(num_features), inplace=True)

        # Variational ansatz for latent + trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
        qc.compose(ansatz, range(num_features, num_features + num_latent + num_trash), inplace=True)

        # Domain‑wall: flip trash qubits to encode a simple pattern
        for i in range(num_trash):
            qc.x(num_features + num_latent + i)

        # Swap test with auxiliary qubit
        aux = num_features + num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_features + num_latent + i, num_features + num_latent + num_trash + i)
        qc.h(aux)

        # Measurement
        qc.measure(aux, cr[0])

        return qc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run the quantum circuit for a single input vector."""
        return self.qnn.predict(x)

__all__ = ["HybridEstimatorAutoencoder"]
