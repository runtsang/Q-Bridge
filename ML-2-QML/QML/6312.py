"""
Quantum neural network for fraud detection that complements the classical
autoencoder‑based feature extractor.  The circuit encodes latent features
with a RealAmplitudes ansatz and outputs a probability via a single
measurement qubit.  The QNN can be trained with a classical optimizer
(e.g., COBYLA) and executed on a Qiskit simulator or real device.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RY
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  Helper: construct a feature‑map circuit from classical latent vector
# --------------------------------------------------------------------------- #
def _feature_map(latent: np.ndarray) -> QuantumCircuit:
    """
    Encode a 2‑dimensional latent vector into rotation angles on a single qubit.
    For higher‑dimensional latents, repeat the pattern across multiple qubits.
    """
    q = QuantumRegister(latent.size, "q")
    circuit = QuantumCircuit(q)
    for i, angle in enumerate(latent):
        circuit.ry(angle, q[i])
    return circuit


# --------------------------------------------------------------------------- #
#  Quantum neural network wrapper
# --------------------------------------------------------------------------- #
class FraudDetectionQNN:
    """
    A variational quantum neural network that takes a 2‑dimensional latent
    vector from the classical autoencoder and outputs a single probability.
    """
    def __init__(self, latent_dim: int = 2, reps: int = 3):
        self.latent_dim = latent_dim
        self.reps = reps
        self.sampler = StatevectorSampler()
        self._build_qnn()

    def _build_qnn(self) -> None:
        # Build the ansatz
        ansatz = RealAmplitudes(self.latent_dim, reps=self.reps)
        # Build the full circuit: feature map + ansatz + measurement
        self.circuit = QuantumCircuit(self.latent_dim, 1)
        # Feature map placeholder; will be composed in predict()
        self.ansatz = ansatz
        self.circuit.append(ansatz, range(self.latent_dim))
        self.circuit.measure(0, 0)  # measurement on first qubit

        # Wrap in a SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],            # no classical inputs in this simple example
            weight_params=ansatz.parameters,
            interpret=lambda x: x[0].real,
            output_shape=1,
            sampler=self.sampler,
        )

    def predict(self, latent: np.ndarray) -> float:
        """Return the predicted fraud probability for a given latent vector."""
        # Build feature‑map circuit
        fm_circuit = _feature_map(latent)
        # Compose with the ansatz
        full_circuit = QuantumCircuit(self.latent_dim, 1)
        full_circuit.compose(fm_circuit, inplace=True)
        full_circuit.compose(self.ansatz, inplace=True)
        full_circuit.measure(0, 0)
        # Execute via SamplerQNN
        res = self.qnn.predict([], weight_vector=full_circuit.parameters)
        return float(res[0])

    def train(
        self,
        latent_data: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 1e-2,
        optimizer_cls=COBYLA,
    ) -> list[float]:
        """Train the QNN to minimize binary cross‑entropy."""
        optimizer = optimizer_cls()
        history: list[float] = []

        for epoch in range(epochs):
            loss = 0.0
            for x, y in zip(latent_data, labels):
                pred = self.predict(x)
                loss += - (y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
            loss /= len(latent_data)
            optimizer.step(lambda _: loss, self.ansatz.parameters)
            history.append(loss)
        return history


__all__ = ["FraudDetectionQNN"]
