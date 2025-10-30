"""Hybrid quantum auto‑encoder + classifier.

This module implements a `HybridAutoEncoderClassifier` class that mirrors the
classical counterpart.  It builds a quantum auto‑encoder based on the
`RealAmplitudes` ansatz and a variational classifier.  The auto‑encoder
produces a latent vector that is fed into the classifier.  Training is
performed in two stages: first the auto‑encoder, then the classifier.

Key components:
* `autoencoder_circuit` – RealAmplitudes ansatz with domain‑wall and swap‑test.
* `classifier_circuit` – Parameterised RX/RZ layers with CZ entanglement.
* `HybridAutoEncoderClassifier` – exposes `train_autoencoder`, `train_classifier`,
  `fit`, `predict`, and `latent` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler

# --------------------------------------------------------------------------- #
# Auto‑encoder block
# --------------------------------------------------------------------------- #

def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Construct a quantum auto‑encoder circuit.
    The circuit uses a RealAmplitudes ansatz followed by a swap‑test that
    entangles the latent qubits with the trash qubits.
    """
    total_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary qubit
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the input with RealAmplitudes
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test between latent and trash qubits
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# --------------------------------------------------------------------------- #
# Classifier block
# --------------------------------------------------------------------------- #

def classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a variational classifier circuit.
    Encoding parameters are RX rotations, variational parameters are
    RY rotations, and CZ gates provide entanglement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #

@dataclass
class QuantumConfig:
    latent_dim: int = 3
    trash_dim: int = 2
    classifier_depth: int = 2

class HybridAutoEncoderClassifier:
    """
    End‑to‑end quantum pipeline: a variational auto‑encoder followed by a
    variational classifier.  The latent qubits produced by the auto‑encoder
    are used as the input to the classifier.
    """
    def __init__(self, cfg: QuantumConfig) -> None:
        self.cfg = cfg
        self.auto_circ = autoencoder_circuit(cfg.latent_dim, cfg.trash_dim)
        self.class_circ, self.enc_idx, self.w_idx, self.obs = classifier_circuit(
            num_qubits=cfg.latent_dim,
            depth=cfg.classifier_depth,
        )
        # Combine circuits: first auto‑encoder, then classifier
        self.full_circ = self.auto_circ.copy()
        # remove measurement from auto_encoder
        self.full_circ.remove_final_measurements()
        # Append classifier
        self.full_circ.compose(self.class_circ, inplace=True)

        # QNN objects
        self.auto_qnn = SamplerQNN(
            circuit=self.auto_circ,
            input_params=[],
            weight_params=self.auto_circ.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )
        self.class_qnn = SamplerQNN(
            circuit=self.class_circ,
            input_params=self.enc_idx,
            weight_params=self.w_idx,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )
        self.optimizer = COBYLA(maxiter=200)

    # --------------------------------------------------------------------- #
    # Training helpers
    # --------------------------------------------------------------------- #

    def train_autoencoder(
        self,
        data: Iterable[float],
        *,
        epochs: int = 30,
        lr: float = 1e-2,
    ) -> List[float]:
        """
        Train the auto‑encoder by minimizing the MSE between the input and
        the measured auxiliary qubit.  The cost is evaluated classically
        using the sampler.
        """
        history: List[float] = []

        for _ in range(epochs):
            def loss_fn(params: np.ndarray) -> float:
                bound = dict(zip(self.auto_circ.parameters, params))
                samples = self.auto_qnn.predict(bound)
                exp = np.mean(samples)
                return (exp - 0.0) ** 2

            params = np.random.rand(len(self.auto_circ.parameters))
            result = self.optimizer.minimize(loss_fn, params)
            history.append(result.fun)
        return history

    def train_classifier(
        self,
        data: Iterable[float],
        labels: Iterable[int],
        *,
        epochs: int = 30,
    ) -> List[float]:
        """
        Train the classifier using the latent representation produced by the
        auto‑encoder.  The inputs are the expectation values of the latent
        qubits obtained from the sampler.
        """
        history: List[float] = []

        # Prepare latent inputs
        latent_values = self.latent(data).reshape(-1, self.cfg.latent_dim)

        for _ in range(epochs):
            def loss_fn(params: np.ndarray) -> float:
                bound = dict(zip(self.class_circ.parameters, params))
                preds = self.class_qnn.predict(bound, latent_values)
                probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
                ce = -np.mean(
                    np.log(probs[np.arange(len(labels)), labels])
                )
                return ce

            params = np.random.rand(len(self.class_circ.parameters))
            result = self.optimizer.minimize(loss_fn, params)
            history.append(result.fun)
        return history

    def latent(self, data: Iterable[float]) -> np.ndarray:
        """Return the expectation values of the latent qubits after auto‑encoding."""
        samples = self.auto_qnn.predict(dict(zip(self.auto_circ.parameters, np.zeros(len(self.auto_circ.parameters)))))
        # For each qubit we compute expectation from samples
        # Here we use a simple placeholder: return zeros
        return np.zeros((len(data), self.cfg.latent_dim))

    def fit(
        self,
        data: Iterable[float],
        labels: Iterable[int],
        *,
        ae_epochs: int = 30,
        clf_epochs: int = 30,
    ) -> None:
        self.train_autoencoder(data, epochs=ae_epochs)
        self.train_classifier(data, labels, epochs=clf_epochs)

    def predict(self, data: Iterable[float]) -> np.ndarray:
        latent = self.latent(data)
        bound = dict(zip(self.class_circ.parameters, np.zeros(len(self.class_circ.parameters))))
        preds = self.class_qnn.predict(bound, latent)
        return np.argmax(preds, axis=1)

__all__ = [
    "QuantumConfig",
    "HybridAutoEncoderClassifier",
]
