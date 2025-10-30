from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class FraudDetectionModel:
    """
    Quantum neural network that takes the latent feature vector produced by
    the classical auto‑encoder and outputs a fraud probability.
    """
    def __init__(self,
                 num_features: int,
                 num_qubits: int | None = None,
                 reps: int = 3,
                 seed: int = 42) -> None:
        algorithm_globals.random_seed = seed
        self.num_features = num_features
        self.num_qubits = num_qubits or num_features
        self.circuit = self._build_circuit(reps)
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(max_trials=200)

    def _build_circuit(self, reps: int) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Feature encoding: rotate each qubit by the corresponding latent value
        for i in range(min(self.num_features, self.num_qubits)):
            circuit.ry(self.num_features * 0.1, qr[i])  # simple scaling

        # Variational ansatz
        ansatz = RealAmplitudes(self.num_qubits, reps=reps)
        circuit.compose(ansatz, qr, inplace=True)

        circuit.measure(qr[0], cr[0])
        return circuit

    def forward(self, latent: np.ndarray) -> float:
        """
        Evaluate the circuit for a single latent vector and return
        the probability of the |1⟩ measurement outcome.
        """
        param_dict = {p: latent[i] for i, p in enumerate(self.circuit.parameters)}
        result = self.sampler.run(self.circuit, parameter_binds=[param_dict]).result()
        counts = result.get_counts()
        prob = counts.get("1", 0) / sum(counts.values())
        return prob

    def train(self,
              latent_vectors: np.ndarray,
              labels: np.ndarray) -> list[float]:
        """
        Train the quantum neural network using the COBYLA optimizer.
        """
        history: list[float] = []
        for _ in range(self.optimizer.max_trials):
            params = self.optimizer.get_parameter_values()
            loss = 0.0
            for vec, lbl in zip(latent_vectors, labels):
                prob = self.forward(vec)
                loss += -np.log(prob) if lbl == 1 else -np.log(1 - prob)
            loss /= len(latent_vectors)
            history.append(loss)
            self.optimizer.step()
            if self.optimizer.converged:
                break
        return history

__all__ = ["FraudDetectionModel"]
