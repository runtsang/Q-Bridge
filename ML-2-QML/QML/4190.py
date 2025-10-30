"""Hybrid quantum autoencoder that mirrors the classical AutoencoderHybrid.

The circuit encodes the input into a latent subspace using a RealAmplitudes
ansatz, extracts a latent vector via a swap‑test with trash qubits, and
provides a SamplerQNN interface for end‑to‑end optimisation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import Aer


class AutoencoderHybrid:
    """Quantum implementation of the hybrid autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of classical features per sample.
    latent_dim : int
        Number of qubits used to store the latent representation.
    num_trash : int
        Number of ancillary qubits used for the swap‑test.
    reps : int
        Number of repetitions for the RealAmplitudes ansatz.
    shots : int
        Number of shots for the sampler.
    backend : Backend | None
        Qiskit backend; defaults to Aer qasm simulator.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 3,
        shots: int = 1024,
        backend=None,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.shots = shots
        self.backend = backend or self._default_backend()
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    @staticmethod
    def _default_backend():
        return Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> QuantumCircuit:
        """Build the hybrid circuit used for encoding and decoding."""
        total_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Feature embedding – embed the first `latent_dim+num_trash` bits
        # of the input into the circuit using a RawFeatureVector.
        feature_map = RawFeatureVector(num_qubits=self.latent_dim + self.num_trash)
        circuit.compose(
            feature_map, range(self.latent_dim + self.num_trash), inplace=True
        )

        # Variational ansatz over the latent+trash qubits
        ansatz = RealAmplitudes(
            num_qubits=self.latent_dim + self.num_trash, reps=self.reps
        )
        circuit.compose(ansatz, range(self.latent_dim + self.num_trash), inplace=True)

        # Swap‑test using an auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Run the sampler and return the probability of measuring |0>.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, input_dim). Only the first
            ``latent_dim + num_trash`` features are used in the feature map.
        """
        # Bind parameters to the circuit – all are static in this example
        param_binds = [{} for _ in range(len(data))]
        result = self.sampler.run(
            self.circuit, shots=self.shots, parameter_binds=param_binds
        )
        counts = result.get_counts()
        p0 = counts.get("0", 0) / self.shots
        p1 = counts.get("1", 0) / self.shots
        return np.array([p0, p1])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode the latent vector back to a classical array.

        In this minimal implementation the decoder is a trivial linear
        mapping; users may replace it with a more sophisticated quantum
        circuit if desired.
        """
        return latent  # identity mapping for demonstration

    def forward(self, data: np.ndarray) -> np.ndarray:
        """End‑to‑end quantum autoencoder."""
        latent = self.encode(data)
        return self.decode(latent)

    def __repr__(self) -> str:
        return (
            f"AutoencoderHybrid(input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"num_trash={self.num_trash}, reps={self.reps}, shots={self.shots})"
        )


__all__ = ["AutoencoderHybrid"]
