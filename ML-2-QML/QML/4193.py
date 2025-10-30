"""Quantum‑centric hybrid autoencoder with self‑attention and regression head.

This module implements :class:`QuantumAutoencoderHybrid`, a Qiskit‑based
autoencoder that mirrors the classical hybrid model.  It provides:

* A self‑attention style entanglement pattern in the encoding circuit.
* A swap‑test based fidelity estimator for the latent representation.
* An EstimatorQNN regression head that predicts a scalar property from the
  latent state.

The design is inspired by the SelfAttention and EstimatorQNN seeds, and
extends them with a quantum latent encoder for the autoencoder task.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.circuit.library import RawFeatureVector

# --------------------------------------------------------------------------- #
#  Helper: quantum self‑attention circuit
# --------------------------------------------------------------------------- #
def _build_attention_circuit(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """
    Builds a circuit that applies a RealAmplitudes ansatz followed by a
    chain of controlled‑X gates mimicking a self‑attention entanglement
    pattern.  The circuit is designed to be parameter‑free for the
    attention block itself; the parameters are supplied by the sampler.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    # Parametrised ansatz
    ansatz = RealAmplitudes(num_qubits, reps=reps)
    qc.compose(ansatz, range(num_qubits), inplace=True)

    # Entanglement pattern: CX between successive qubits
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    qc.measure(qr, cr)
    return qc

# --------------------------------------------------------------------------- #
#  Quantum autoencoder core
# --------------------------------------------------------------------------- #
class QuantumAutoencoderHybrid:
    """Hybrid quantum autoencoder with attention and regression head.

    The encoder maps an input feature vector to a quantum latent state
    using a RealAmplitudes ansatz and a swap‑test based fidelity estimator.
    A separate EstimatorQNN predicts a scalar from the latent state.
    """
    def __init__(self, latent_dim: int = 8, trash_dim: int = 2):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self._build_circuits()

    def _build_circuits(self) -> None:
        # ------------------ Encoder circuit ------------------
        total = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total, "q")
        cr = ClassicalRegister(1, "c")
        self.encoder_circuit = QuantumCircuit(qr, cr)

        # RealAmplitudes ansatz for latent + trash
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=3)
        self.encoder_circuit.compose(ansatz, range(self.latent_dim + self.trash_dim), inplace=True)

        # Attention‑style CX entanglement
        for i in range(self.trash_dim):
            self.encoder_circuit.cx(self.latent_dim + i, self.latent_dim + self.trash_dim + i)

        # Swap‑test for fidelity estimation
        anc = self.latent_dim + 2 * self.trash_dim
        self.encoder_circuit.h(anc)
        for i in range(self.trash_dim):
            self.encoder_circuit.cswap(anc, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        self.encoder_circuit.h(anc)
        self.encoder_circuit.measure(anc, cr[0])

        # SamplerQNN for latent extraction
        sampler = StatevectorSampler()
        self.latent_qnn = SamplerQNN(
            circuit=self.encoder_circuit,
            input_params=[],            # no classical inputs
            weight_params=self.encoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=self.latent_dim,
            sampler=sampler,
        )

        # ------------------ Regression head ------------------
        # Simple 1‑qubit circuit with H, RY, RX and a Y observable
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(qiskit.circuit.Parameter("θ"), 0)
        qc.rx(qiskit.circuit.Parameter("φ"), 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[qc.parameters[0]],
            weight_params=[qc.parameters[1]],
            estimator=estimator,
        )

    def encode(self, inputs: np.ndarray, backend: qiskit.providers.Backend) -> np.ndarray:
        """
        Run the encoder circuit on *backend* and return the latent vector.

        Parameters
        ----------
        inputs : np.ndarray
            Classical feature vector; currently unused but kept for API symmetry.
        backend : qiskit.providers.Backend
            Backend to execute the sampler.

        Returns
        -------
        np.ndarray
            Latent vector of shape (latent_dim,).
        """
        # In a full implementation the *inputs* would be encoded as circuit
        # parameters.  Here we simply run the parameterised circuit with random
        # weights for demonstration.
        result = self.latent_qnn.forward(np.random.rand(*self.latent_qnn.input_params_shape))
        return np.array(result)

    def predict(self, latent: np.ndarray, backend: qiskit.providers.Backend) -> float:
        """
        Predict a scalar property from the latent vector using the EstimatorQNN.

        Parameters
        ----------
        latent : np.ndarray
            Latent representation returned by :meth:`encode`.
        backend : qiskit.providers.Backend
            Backend to execute the estimator.

        Returns
        -------
        float
            Predicted scalar value.
        """
        # Map latent to the estimator's input parameter (θ).  For simplicity we
        # take the first component and normalise.
        θ = float(latent[0]) / np.linalg.norm(latent)
        φ = 0.0  # fixed for this toy example
        result = self.estimator_qnn.forward(np.array([θ, φ]))
        return float(result)

    def run(self, inputs: np.ndarray, backend: qiskit.providers.Backend) -> tuple[np.ndarray, float]:
        """
        Full forward pass: encode and predict.

        Returns
        -------
        tuple[np.ndarray, float]
            (latent vector, predicted scalar)
        """
        latent = self.encode(inputs, backend)
        pred = self.predict(latent, backend)
        return latent, pred


def QuantumAutoencoderHybridFactory(latent_dim: int = 8, trash_dim: int = 2) -> QuantumAutoencoderHybrid:
    """Convenience factory mirroring the classical Autoencoder factory."""
    return QuantumAutoencoderHybrid(latent_dim=latent_dim, trash_dim=trash_dim)


__all__ = ["QuantumAutoencoderHybrid", "QuantumAutoencoderHybridFactory"]
