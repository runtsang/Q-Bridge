"""AutoencoderGen: a quantum variational autoencoder with a hybrid training loop."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderGen:
    """Quantum variational autoencoder that learns a latent distribution over a set of qubits."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5) -> None:
        algorithm_globals.random_seed = 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(maxiter=200)

    def _build_circuit(self) -> QuantumCircuit:
        """Builds a variational circuit with a swap‑test for reconstruction."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz for latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def sample_latent(self, num_samples: int = 1000) -> np.ndarray:
        """Samples the latent distribution by running the circuit."""
        shots = num_samples
        result = self.sampler.run(self.circuit, shots=shots).result()
        counts = result.get_counts()
        # Convert counts to probabilities over the auxiliary qubit
        probs = np.array([counts.get("0", 0), counts.get("1", 0)]) / shots
        return probs

    def reconstruction_loss(self, target: np.ndarray) -> float:
        """Computes a simple fidelity‑based loss between target and sampled latent."""
        probs = self.sample_latent(num_samples=target.shape[0])
        # Target is assumed to be a binary distribution over the auxiliary qubit
        return np.mean((probs - target) ** 2)

    def train(self, data: np.ndarray, *, epochs: int = 100, lr: float = 1e-3) -> list[float]:
        """Hybrid training loop that updates the circuit parameters."""
        loss_history: list[float] = []
        for epoch in range(epochs):
            def objective(theta: np.ndarray) -> float:
                # Update circuit parameters
                param_dict = dict(zip(self.circuit.parameters, theta))
                self.circuit.assign_parameters(param_dict, inplace=True)
                return self.reconstruction_loss(data)
            result = self.optimizer.optimize(num_vars=len(self.circuit.parameters), objective_function=objective)
            loss_history.append(result.fun)
        return loss_history

__all__ = ["AutoencoderGen"]
