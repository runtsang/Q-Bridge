from __future__ import annotations
import numpy as np
from typing import Iterable
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def _identity_interpret(x):
    """No‑op interpret: pass through raw output."""
    return x

def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build the swap‑test based auto‑encoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

class HybridAutoencoder:
    """Quantum‑centric auto‑encoder wrapper exposing a ``run`` method."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        shots: int = 1024,
    ) -> None:
        self.circuit = _autoencoder_circuit(num_latent, num_trash)
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=_identity_interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the quantum circuit with the supplied weight parameters."""
        return self.qnn(thetas)

# Classical stand‑in that mimics the quantum layer for CPU‑only runs
class ClassicalFCL:
    """A lightweight neural layer that emulates a parameterised quantum circuit."""
    def __init__(self, n_features: int = 1) -> None:
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

__all__ = ["HybridAutoencoder", "ClassicalFCL"]
