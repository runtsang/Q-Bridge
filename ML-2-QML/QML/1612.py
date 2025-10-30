"""Quantum autoencoder using Qiskit Machine Learning.

The class builds a variational circuit with:
- RealAmplitudes feature map for input data.
- Encoder and decoder ansatzes.
- Swap‑test to compute reconstruction fidelity.
- Training via SPSA with optional early stopping.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from qiskit import Aer
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, SwapGate
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import SPSA
from qiskit.primitives import Sampler as SamplerPrimitive


def _as_tensor(data: Iterable[float]) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


class Autoencoder:
    """Variational quantum autoencoder.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    latent_dim : int, default 3
        Number of latent qubits.
    reps : int, default 3
        Number of repetitions for the ansatzes.
    backend : qiskit.providers.Backend, optional
        Simulation backend; defaults to state‑vector simulator.
    """

    def __init__(
        self,
        num_features: int,
        *,
        latent_dim: int = 3,
        reps: int = 3,
        backend=None,
    ) -> None:
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.reps = reps
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.sampler = SamplerPrimitive(backend=self.backend)
        self.qnn = self._build_qnn()

    def _feature_map(self, qreg: QuantumRegister) -> QuantumCircuit:
        """RealAmplitudes feature embedding for input data."""
        fmap = RealAmplitudes(self.num_features, reps=self.reps)
        fmap = fmap.assign_parameters(np.arange(self.num_features) / self.num_features, qreg)
        return fmap

    def _ansatz(self, qreg: QuantumRegister, name: str) -> QuantumCircuit:
        """Variational encoder or decoder ansatz."""
        return RealAmplitudes(len(qreg), reps=self.reps, name=name)

    def _swap_test(self, circuit: QuantumCircuit, aux: int, target: int) -> None:
        """Insert swap‑test to compare ancilla and target."""
        circuit.h(aux)
        for i in range(target):
            circuit.cswap(aux, i, i + target)
        circuit.h(aux)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full autoencoder circuit."""
        n_q = self.latent_dim + 2 * self.num_features + 1  # latent + trash + ancilla
        qr = QuantumRegister(n_q, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature embedding
        qc.append(self._feature_map(qr[: self.num_features]), range(self.num_features))

        # Encoder
        encoder = self._ansatz(qr[self.num_features : self.num_features + self.latent_dim], "enc")
        qc.append(encoder, range(self.num_features, self.num_features + self.latent_dim))

        # Trash qubits for disentanglement
        qc.barrier()
        trash_start = self.num_features + self.latent_dim
        trash_end = trash_start + self.num_features
        qc.append(
            self._ansatz(qr[trash_start:trash_end], "trash_enc"),
            range(trash_start, trash_end),
        )

        # Swap test between latent and trash
        aux = n_q - 1
        self._swap_test(qc, aux, self.num_features)

        # Decoder mirrors encoder
        decoder = self._ansatz(qr[trash_start:trash_end], "dec")
        qc.append(decoder, range(trash_start, trash_end))

        qc.append(
            self._ansatz(qr[self.num_features : self.num_features + self.latent_dim], "dec_latent"),
            range(self.num_features, self.num_features + self.latent_dim),
        )

        qc.measure(aux, cr[0])
        return qc

    def _build_qnn(self) -> SamplerQNN:
        """Wrap the circuit in a SamplerQNN for training."""
        circuit = self._build_circuit()
        weight_params = [p for p in circuit.parameters if "enc" in p.name or "dec" in p.name]
        return SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=weight_params,
            interpret=lambda x: float(np.mean(x)),
            output_shape=(1,),
            sampler=self.sampler,
        )

    def fidelity_loss(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss as 1 - mean fidelity over batch."""
        return 1.0 - np.mean(np.abs(outputs - targets))

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 0.1,
        early_stop_patience: int | None = None,
        batch_size: int = 32,
    ) -> Tuple[list[float], list[float]]:
        """Train the quantum autoencoder.

        Returns
        -------
        train_loss : list[float]
            Training loss history.
        val_loss : list[float]
            Validation loss history (if early stopping is enabled).
        """
        data = _as_tensor(data)
        n = len(data)
        split = int(0.8 * n)
        train_data, val_data = data[:split], data[split:]

        optimizer = SPSA(self.qnn, lr=lr)

        history = {"train": [], "val": []}
        best_val = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(len(train_data))
            train_loss = 0.0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[idx[i : i + batch_size]]
                # Forward pass
                outputs = self.qnn(batch, return_grad=False)
                loss = self.fidelity_loss(outputs, batch)
                # Backward pass
                optimizer.step(loss)
                train_loss += loss * len(batch)
            train_loss /= len(train_data)
            history["train"].append(train_loss)

            if early_stop_patience is not None:
                val_loss = 0.0
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i : i + batch_size]
                    outputs = self.qnn(batch, return_grad=False)
                    val_loss += self.fidelity_loss(outputs, batch) * len(batch)
                val_loss /= len(val_data)
                history["val"].append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop_patience:
                    break

        return history["train"], history.get("val", [])

    def encode(self, sample: np.ndarray) -> np.ndarray:
        """Encode a single sample to its latent representation."""
        circuit = self._build_circuit()
        # Replace feature map with the sample
        for i, val in enumerate(sample):
            circuit.assign_parameters([val], circuit.qubits[i])
        result = self.sampler.run(circuit).result()
        return result.get_counts()[0]

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode a latent vector back to data space."""
        # Placeholder: in a full implementation we would construct a decoder circuit.
        return latent  # pragma: no cover


__all__ = ["Autoencoder"]
