"""
Quantum hybrid autoencoder for fraud detection.
Implements a RealAmplitudes ansatz with a swap‑test domain wall
to classify latent representations produced by the classical encoder.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoderFraudDetector:
    """
    Quantum counterpart of the classical HybridAutoencoderFraudDetector.
    Encodes a latent vector into a quantum circuit, applies a RealAmplitudes
    ansatz, then performs a swap‑test with a domain‑wall prepared ancillary
    state to produce a binary classification probability.
    """
    def __init__(self, latent_dim: int, num_trash: int = 2, reps: int = 5):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = Sampler()
        self.qnn = self._build_qnn()

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _domain_wall(self, qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """Apply X gates to the second half of the domain‑wall block."""
        for i in range(start, end):
            qc.x(i)
        return qc

    def _swap_test(self, qc: QuantumCircuit, aux: int, targets: Iterable[int]) -> QuantumCircuit:
        qc.h(aux)
        for t in targets:
            qc.cswap(aux, t, t + self.num_trash)
        qc.h(aux)
        return qc

    def _build_qnn(self) -> SamplerQNN:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent vector into first qubits via Ry rotations
        for i in range(self.latent_dim):
            qc.ry(0.0, i)  # placeholder; will be parameterised externally

        # Domain wall preparation
        dw = QuantumCircuit(self.num_trash)
        dw = self._domain_wall(dw, 0, self.num_trash)
        qc.compose(dw, range(self.latent_dim, self.latent_dim + self.num_trash), inplace=True)

        # Ansatz
        qc.compose(self._ansatz(self.latent_dim + self.num_trash), range(0, self.latent_dim + self.num_trash), inplace=True)

        # Swap test with ancillary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc = self._swap_test(qc, aux, range(self.latent_dim))

        qc.measure(aux, cr[0])

        # Interpret measurement as a probability of fraud
        def interpret(x: np.ndarray) -> np.ndarray:
            return x.astype(float)

        return SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def predict(self, latent: np.ndarray) -> float:
        """
        Return fraud probability for a latent vector.
        The latent vector is expected to be a 1‑D array of length `latent_dim`.
        """
        # Map latent values to rotation angles (simple scaling)
        angles = 2 * np.pi * (latent - latent.min()) / (latent.max() - latent.min() + 1e-8)
        # Build parameter dictionary
        param_dict = {param: angle for param, angle in zip(self.qnn.weight_params, angles)}
        # Execute circuit
        result = self.sampler.run(self.qnn.circuit, param_dict).result()
        probs = result.get_counts()
        # Convert counts to probability of fraud (class 1)
        fraud_prob = probs.get('1', 0) / sum(probs.values())
        return fraud_prob

__all__ = ["HybridAutoencoderFraudDetector"]
