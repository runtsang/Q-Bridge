"""
Variational quantum autoencoder (VQA) that mirrors the
``AutoencoderModel`` interface.  It uses Qiskit’s
``SamplerQNN`` to produce latent vectors from quantum
measurements and reconstructs inputs via a second circuit.
The same class name is shared with the classical implementation so
researchers can swap seamlessly between the two paradigms.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _embed_statevector(features: np.ndarray) -> QuantumCircuit:
    """Amplitude encode a real vector into a quantum state."""
    n_qubits = int(np.ceil(np.log2(len(features))))
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    # Pad with zeros if necessary
    padded = np.pad(features, (0, 2**n_qubits - len(features)), "constant")
    vec = padded / np.linalg.norm(padded)
    sv = Statevector(vec)
    qc.initialize(sv.data, qr)
    return qc


def _measure_latent(qc: QuantumCircuit, num_latent: int) -> List[int]:
    """Measure the first ``num_latent`` qubits and return classical bits."""
    cr = ClassicalRegister(num_latent)
    qc.add_register(cr)
    qc.measure(range(num_latent), cr)
    return cr


# --------------------------------------------------------------------------- #
# Quantum autoencoder class
# --------------------------------------------------------------------------- #
class AutoencoderModel:
    """
    Variational quantum autoencoder with a dual‑circuit architecture.
    ``encode`` projects classical data into a quantum latent state
    and measures a subset of qubits.  ``decode`` takes a latent
    vector, re‑prepares a quantum state, runs a decoder circuit,
    and samples the original feature distribution.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        hidden_layers: Tuple[int,...] = (4, 4),
        reps: int = 3,
        sampler: Sampler | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.reps = reps
        self.sampler = sampler or Sampler()

        # Encoder circuit
        self._build_encoder()
        # Decoder circuit
        self._build_decoder()

    # --------------------------------------------------------------------- #
    # Circuit construction
    # --------------------------------------------------------------------- #
    def _build_encoder(self) -> None:
        n_qubits = self.latent_dim
        self.ansatz = TwoLocal(
            n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement="full",
            reps=self.reps,
            parameter_prefix="enc",
        )
        self.encoder_qc = QuantumCircuit(n_qubits)
        self.encoder_qc.append(self.ansatz, range(n_qubits))

        # Latent measurement
        self.latent_cr = ClassicalRegister(self.latent_dim)
        self.encoder_qc.add_register(self.latent_cr)
        self.encoder_qc.measure(range(self.latent_dim), self.latent_cr)

        # Wrap in SamplerQNN
        self.encoder_qnn = SamplerQNN(
            circuit=self.encoder_qc,
            input_params=[],
            weight_params=self.ansatz.parameters,
            interpret=lambda x: x,
            output_shape=(self.latent_dim,),
            sampler=self.sampler,
        )

    def _build_decoder(self) -> None:
        n_qubits = self.input_dim
        self.decoder_ansatz = TwoLocal(
            n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement="full",
            reps=self.reps,
            parameter_prefix="dec",
        )
        self.decoder_qc = QuantumCircuit(n_qubits)
        self.decoder_qc.append(self.decoder_ansatz, range(n_qubits))

        # Output sampling
        self.decoder_qnn = SamplerQNN(
            circuit=self.decoder_qc,
            input_params=[],
            weight_params=self.decoder_ansatz.parameters,
            interpret=lambda x: x,
            output_shape=(self.input_dim,),
            sampler=self.sampler,
        )

    # --------------------------------------------------------------------- #
    # Forward passes
    # --------------------------------------------------------------------- #
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode a classical feature vector into a latent vector.
        Parameters
        ----------
        features : np.ndarray, shape (input_dim,)
            Normalised real‑valued input.
        Returns
        -------
        latent : np.ndarray, shape (latent_dim,)
            Probabilistic measurement outcomes (0/1).
        """
        if features.ndim!= 1 or len(features)!= self.input_dim:
            raise ValueError("Input must be a vector of length ``input_dim``.")
        # Prepare state and run
        enc_features = _embed_statevector(features)
        qc = enc_features + self.encoder_qc
        result = self.sampler.run(qc, shots=1).result()
        counts = result.get_counts()
        # Convert counts to bits
        bits = list(counts.keys())[0][::-1]  # reverse order to match qubit indices
        latent = np.array([int(b) for b in bits[: self.latent_dim]], dtype=float)
        return latent

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode a latent vector back to the feature space.
        Parameters
        ----------
        latent : np.ndarray, shape (latent_dim,)
            Binary latent vector.
        Returns
        -------
        recon : np.ndarray, shape (input_dim,)
            Reconstructed feature probabilities.
        """
        if latent.ndim!= 1 or len(latent)!= self.latent_dim:
            raise ValueError("Latent must be a vector of length ``latent_dim``.")
        # Prepare a circuit with the latent as initial state
        init_qc = QuantumCircuit(self.latent_dim)
        for idx, bit in enumerate(latent):
            if bit:
                init_qc.x(idx)
        qc = init_qc + self.decoder_qc
        result = self.sampler.run(qc, shots=1).result()
        counts = result.get_counts()
        # Probability distribution over basis states
        probs = np.zeros(self.input_dim)
        for state, count in counts.items():
            idx = int(state[::-1], 2)
            probs[idx] = count
        probs /= probs.sum()
        return probs

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Full autoencoding: encode → decode."""
        latent = self.encode(features)
        return self.decode(latent)

    # --------------------------------------------------------------------- #
    # Hybrid training loop
    # --------------------------------------------------------------------- #
    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> List[float]:
        """
        Train both encoder and decoder parameters jointly using a
        classical optimiser over the negative log‑likelihood of the
        reconstructed probability distribution.
        """
        # Flatten parameters into a single vector
        enc_params = np.array([float(p) for p in self.ansatz.parameters])
        dec_params = np.array([float(p) for p in self.decoder_ansatz.parameters])
        all_params = np.concatenate([enc_params, dec_params])

        n_params = len(all_params)
        history: List[float] = []

        def loss_fn(params: np.ndarray) -> float:
            # Split parameters
            self.ansatz.assign_parameters(params[: len(enc_params)])
            self.decoder_ansatz.assign_parameters(params[len(enc_params) :])
            # Re‑build QNNs with new weights
            self.encoder_qnn.weight_params = self.ansatz.parameters
            self.decoder_qnn.weight_params = self.decoder_ansatz.parameters
            # Compute reconstruction loss over the dataset
            losses = []
            for x in data:
                recon = self.forward(x)
                # Negative log‑likelihood assuming Bernoulli distribution
                loss = -np.sum(x * np.log(recon + 1e-12) + (1 - x) * np.log(1 - recon + 1e-12))
                losses.append(loss)
            return np.mean(losses)

        # Simple gradient descent
        for epoch in range(epochs):
            grad = np.zeros_like(all_params)
            eps = 1e-6
            for i in range(n_params):
                perturbed = all_params.copy()
                perturbed[i] += eps
                grad[i] = (loss_fn(perturbed) - loss_fn(all_params)) / eps
            all_params -= lr * grad
            history.append(loss_fn(all_params))
        return history


__all__ = [
    "AutoencoderModel",
]
