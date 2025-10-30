from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

class AutoencoderHybrid:
    """Quantum implementation of a hybrid auto‑encoder.

    The encoder is a parameterised RealAmplitudes ansatz that maps an
    input vector to a latent sub‑space.  The decoder uses a swap‑test
    based measurement to reconstruct the input from the latent
    registers.  Auxiliary SamplerQNN and EstimatorQNN circuits are
    attached to impose a prior on the latent distribution and a
    regression constraint, mirroring the classical surrogate nets.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.seed = seed

        # Encoder ansatz
        self.encoder_circuit = RealAmplitudes(
            num_qubits=latent_dim + num_trash, reps=reps, seed=seed
        )

        # Decoder swap‑test circuit
        self.decoder_circuit = self._build_decoder()

        # QNN wrappers
        self.sampler_qnn = SamplerQNN(
            circuit=self.decoder_circuit,
            input_params=[],
            weight_params=self.decoder_circuit.parameters,
            sampler=StatevectorSampler(),
            interpret=lambda x: x,
            output_shape=2,
        )
        self.estimator_qnn = EstimatorQNN(
            circuit=self.decoder_circuit,
            input_params=[],
            weight_params=self.decoder_circuit.parameters,
            estimator=StatevectorEstimator(),
            observables=[("Z", 1)],
        )

    # ------------------------------------------------------------------
    def _build_decoder(self) -> QuantumCircuit:
        """Construct the swap‑test based decoder."""
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Apply encoder ansatz on latent + trash qubits
        qc.append(self.encoder_circuit, range(0, self.latent_dim + self.num_trash))

        # Swap‑test
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(qr[aux])
        for i in range(self.num_trash):
            qc.cswap(qr[aux], qr[self.latent_dim + i], qr[self.latent_dim + self.num_trash + i])
        qc.h(qr[aux])

        # Measurement
        qc.measure(qr[aux], cr[0])
        return qc

    # ------------------------------------------------------------------
    def encode(self, input_vector: np.ndarray) -> np.ndarray:
        """Return a sample from the latent distribution using the sampler."""
        qc = self.encoder_circuit.copy()
        if len(input_vector)!= self.input_dim:
            raise ValueError("Input vector length mismatch.")
        # Map classical input values to Ry angles on the first qubits
        for idx, val in enumerate(input_vector):
            qc.ry(val, idx % (self.latent_dim + self.num_trash))
        samp = StatevectorSampler()
        result = samp.run(qc).result().get_counts()
        return np.array(list(result.values()), dtype=float)

    # ------------------------------------------------------------------
    def decode(self, latent_sample: np.ndarray) -> np.ndarray:
        """Reconstruct an input vector from a latent sample using the swap‑test."""
        qc = self.decoder_circuit.copy()
        for idx, val in enumerate(latent_sample):
            qc.ry(val, idx % qc.num_qubits)
        samp = StatevectorSampler()
        result = samp.run(qc).result().get_counts()
        return np.array(list(result.values()), dtype=float)

    # ------------------------------------------------------------------
    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        learning_rate: float = 0.01,
        optimizer_cls=COBYLA,
        **opt_kwargs,
    ) -> list[float]:
        """Train the encoder parameters and the auxiliary QNNs."""
        history: list[float] = []
        optimizer = optimizer_cls(
            params=list(self.encoder_circuit.parameters) +
                   list(self.decoder_circuit.parameters),
            **opt_kwargs,
        )

        for _ in range(epochs):
            loss = 0.0
            for x in data:
                # Encode
                x_qc = self.encoder_circuit.copy()
                for i, val in enumerate(x):
                    x_qc.ry(val, i % (self.latent_dim + self.num_trash))
                samp = StatevectorSampler()
                latent_counts = samp.run(x_qc).result().get_counts()

                # Decode
                decoded_counts = self.decode(list(latent_counts.keys())[0])

                # Reconstruction loss (Hamming distance)
                recon_err = np.sum(np.abs(decoded_counts - x))
                loss += recon_err

                # Latent regularisation via sampler_qnn
                samp_loss = self.sampler_qnn.forward(
                    torch.tensor(list(latent_counts.values()), dtype=torch.float32)
                )
                loss += samp_loss.mean().item()

                # Estimator regularisation
                est_loss = self.estimator_qnn.forward(
                    torch.tensor(list(latent_counts.values()), dtype=torch.float32)
                )
                loss += est_loss.mean().item()

            loss /= len(data)
            history.append(loss)

            # Update parameters
            optimizer.step()
        return history

__all__ = ["AutoencoderHybrid"]
