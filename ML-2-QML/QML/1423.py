"""Quantum autoencoder using a RealAmplitudes ansatz, swap‑test latent read‑out, and a COBYLA/Adam training loop."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Callable, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as SamplerPrimitive
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


def quantum_autoencoder_circuit(
    num_latent: int,
    num_trash: int,
    reps: int = 5,
) -> QuantumCircuit:
    """Build a variational circuit with a swap‑test based latent read‑out."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz on latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply X gates to create a domain wall between qubits start and end."""
    for i in range(start, end):
        circuit.x(i)
    return circuit


def build_qnn(
    num_latent: int,
    num_trash: int,
    sampler: SamplerPrimitive | None = None,
    interpret: Callable[[np.ndarray], np.ndarray] | None = None,
) -> SamplerQNN:
    """Instantiate a SamplerQNN with the quantum autoencoder circuit."""
    if sampler is None:
        sampler = SamplerPrimitive()
    qc = quantum_autoencoder_circuit(num_latent, num_trash)
    if interpret is None:
        interpret = lambda x: x  # identity

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


class QuantumAutoencoder:
    """Wrapper that trains a quantum autoencoder using COBYLA or Adam."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 5,
        optimizer: str = "cobyla",
        lr: float = 1e-2,
        max_iter: int = 200,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.optimizer_name = optimizer.lower()
        self.lr = lr
        self.max_iter = max_iter
        self.qnn = build_qnn(num_latent, num_trash)
        self._initialize_optimizer()

    # ------------------------------------------------------------------
    def _initialize_optimizer(self) -> None:
        if self.optimizer_name == "cobyla":
            self.optimizer = COBYLA()
        else:
            # Default to Adam via Qiskit’s variational framework
            self.optimizer = self.qnn.optimizer
            self.optimizer.set_options({"learning_rate": self.lr, "maxiter": self.max_iter})

    # ------------------------------------------------------------------
    def loss_function(self, params: np.ndarray, data: np.ndarray) -> float:
        """Mean squared error between target and measurement probabilities."""
        self.qnn.set_weights(params)
        probs = self.qnn.predict(data)
        # probs shape (batch, 2); we use the probability of measuring 1
        preds = probs[:, 1]
        return float(np.mean((preds - data) ** 2))

    # ------------------------------------------------------------------
    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        """Train the quantum autoencoder and return loss history."""
        history: list[float] = []
        params = np.array(self.qnn.get_weights())

        for epoch in range(epochs):
            # Simple batch loop
            epoch_loss = 0.0
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                loss = self.loss_function(params, batch)
                epoch_loss += loss
                if self.optimizer_name == "cobyla":
                    params = self.optimizer.minimize(
                        lambda p: self.loss_function(p, batch),
                        params,
                        options={"maxiter": 20},
                    )
                else:
                    # Adam gradient step
                    grads = np.gradient(self.loss_function, params)[0]
                    params -= self.lr * grads
            epoch_loss /= (len(data) // batch_size)
            history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs} loss: {epoch_loss:.6f}")
        self.qnn.set_weights(params)
        return history


__all__ = ["quantum_autoencoder_circuit", "domain_wall", "build_qnn", "QuantumAutoencoder"]
