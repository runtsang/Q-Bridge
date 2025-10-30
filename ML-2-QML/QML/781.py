"""Quantum autoencoder implemented with Qiskit and variational circuits."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals

class AutoencoderHybrid:
    """A quantum autoencoder that learns a latent representation via a variational circuit."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 3, seed: int = 42):
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the variational autoencoder circuit."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode: variational circuit on latent + trash qubits
        var_circ = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        circuit.compose(var_circ, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap test to compare latent states
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(qr[aux])
        for i in range(self.num_trash):
            circuit.cx(qr[aux], qr[self.num_latent + i])
            circuit.cx(qr[self.num_latent + i], qr[self.num_trash + i])
            circuit.cx(qr[self.num_trash + i], qr[aux])  # equivalent to cswap
        circuit.h(qr[aux])
        circuit.measure(qr[aux], cr[0])
        return circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum circuit and return the measurement probabilities."""
        result = self.sampler.run(self.circuit)
        counts = result.get_counts()
        probs = {k: v / sum(counts.values()) for k, v in counts.items()}
        return probs

def train_quantum_autoencoder(
    model: AutoencoderHybrid,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    optimizer_cls=COBYLA,
    device=None,
) -> list[float]:
    """Very simple training loop for the quantum autoencoder."""
    opt = optimizer_cls(maxiter=epochs)
    history: list[float] = []

    def loss_func(params):
        # Update circuit parameters
        for name, val in zip(model.circuit.parameters, params):
            name.assign(val)
        # Evaluate reconstruction loss (placeholder)
        probs = model.forward(data)
        # Dummy loss: negative log probability of '0'
        loss = -np.log(probs.get('0', 1e-9))
        return loss

    init_params = np.array([float(p) for p in model.circuit.parameters])
    res = opt.minimize(loss_func, init_params)
    history.append(res.fun)
    return history

__all__ = ["AutoencoderHybrid", "train_quantum_autoencoder"]
