"""Quantum hybrid autoencoder + classifier module.

The quantum implementation uses a RealAmplitudes ansatz for the autoencoder
circuit and a layered variational circuit for classification.  The class
mirrors the classical API but executes all operations on a qiskit
simulator.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


def build_autoencoder_circuit(
    num_latent: int,
    num_trash: int,
) -> Tuple[QuantumCircuit, List[ParameterVector]]:
    """Construct a quantum autoencoder with a swap‑test reconstruction."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit, list(ansatz.parameters)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, List[ParameterVector]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    circuit = QuantumCircuit(num_qubits)

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    return circuit, list(weights)


class HybridAutoEncoderClassifier:
    """Quantum hybrid autoencoder + classifier."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        depth: int,
        seed: int = 42,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.depth = depth
        self.seed = seed

        self.autoencoder_circuit, self.auto_encoder_params = build_autoencoder_circuit(
            num_latent, num_trash
        )
        self.classifier_circuit, self.classifier_params = build_classifier_circuit(
            num_latent, depth
        )

        self.sampler = Sampler()
        self.auto_qnn = SamplerQNN(
            circuit=self.autoencoder_circuit,
            input_params=[],
            weight_params=self.auto_encoder_params,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.classifier_qnn = SamplerQNN(
            circuit=self.classifier_circuit,
            input_params=self.auto_encoder_params,
            weight_params=self.classifier_params,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions for a batch of inputs."""
        # Encode with autoencoder
        encoded = self.auto_qnn.predict(x)
        # Pass latent vector to classifier
        logits = self.classifier_qnn.predict(encoded)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return (probs[:, 1] > threshold).astype(np.int64)

    def train_autoencoder(
        self,
        data: np.ndarray,
        *,
        epochs: int = 30,
        lr: float = 0.01,
        seed: int | None = None,
    ) -> List[float]:
        """Optimize autoencoder parameters with COBYLA."""
        seed = seed or self.seed
        opt = COBYLA(maxiter=epochs)
        history: List[float] = []
        params = np.random.rand(len(self.auto_encoder_params))

        for _ in range(epochs):
            def loss_fn(p):
                self.auto_qnn.set_parameters(p)
                pred = self.auto_qnn.predict(data)
                return np.mean((pred - data) ** 2)

            params = opt.minimize(loss_fn, params, maxiter=1, disp=False)
            loss = loss_fn(params)
            history.append(loss)
        return history

    def train_classifier(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 20,
        lr: float = 0.01,
        seed: int | None = None,
    ) -> List[float]:
        """Optimize classifier parameters with COBYLA."""
        seed = seed or self.seed
        opt = COBYLA(maxiter=epochs)
        history: List[float] = []
        params = np.random.rand(len(self.classifier_params))

        for _ in range(epochs):
            def loss_fn(p):
                self.classifier_qnn.set_parameters(p)
                logits = self.classifier_qnn.predict(data)
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                return -np.mean(probs[range(len(labels)), labels])

            params = opt.minimize(loss_fn, params, maxiter=1, disp=False)
            loss = loss_fn(params)
            history.append(loss)
        return history


__all__ = [
    "build_autoencoder_circuit",
    "build_classifier_circuit",
    "HybridAutoEncoderClassifier",
]
