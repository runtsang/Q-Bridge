"""Hybrid quantum autoencoder: variational encoder + swap‑test kernel."""
from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class QuantumHybridAutoencoder:
    """
    Variational autoencoder in the quantum domain.
    * Encoder: RealAmplitudes ansatz mapping classical input to a latent register.
    * Decoder: Reverse ansatz reconstructing the input.
    * Kernel: Swap‑test between two latent states to evaluate similarity.
    """
    def __init__(self, num_features: int, latent_dim: int, reps: int = 3) -> None:
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.reps = reps
        self.sampler = Sampler()
        algorithm_globals.random_seed = 42

    def _build_encoder(self, circuit: QuantumCircuit, feature_wires: list[int]) -> QuantumCircuit:
        """Append a RealAmplitudes encoder to the circuit."""
        enc = RealAmplitudes(self.num_features, reps=self.reps)
        circuit.compose(enc, qubits=feature_wires, inplace=True)
        return circuit

    def _build_decoder(self, circuit: QuantumCircuit, feature_wires: list[int]) -> QuantumCircuit:
        """Append a RealAmplitudes decoder (same ansatz) to the circuit."""
        dec = RealAmplitudes(self.num_features, reps=self.reps)
        circuit.compose(dec.inverse(), qubits=feature_wires, inplace=True)
        return circuit

    def encode_circuit(self) -> QuantumCircuit:
        """Full circuit that maps input data to a latent state."""
        qr = QuantumRegister(self.latent_dim + self.num_features, name="q")
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)

        # Feature encoding
        self._build_encoder(qc, list(range(self.num_features)))

        # Swap‑test auxiliary qubit
        aux = self.latent_dim + self.num_features
        qc.h(aux)
        for i in range(self.latent_dim):
            qc.cswap(aux, i, self.num_features + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate the kernel value k(x,y) = |⟨ψ_x|ψ_y⟩|^2
        via a swap‑test on two latent states encoded from x and y.
        """
        # Encode both inputs into two separate circuits
        qc_x = self.encode_circuit()
        qc_y = self.encode_circuit()

        # Combine into a single circuit that prepares |ψ_x⟩⊗|ψ_y⟩
        qc = QuantumCircuit(self.latent_dim + self.num_features,
                            self.latent_dim + self.num_features,
                            ClassicalRegister(1), ClassicalRegister(1))
        qc.compose(qc_x, qubits=list(range(self.latent_dim + self.num_features)), inplace=True)
        qc.compose(qc_y, qubits=list(range(self.latent_dim + self.num_features,
                                            2 * (self.latent_dim + self.num_features))), inplace=True)

        # Swap‑test on the two latent blocks
        aux1 = 2 * (self.latent_dim + self.num_features)
        aux2 = aux1 + 1
        qc.h(aux1)
        for i in range(self.latent_dim):
            qc.cswap(aux1, i, self.latent_dim + i)
        qc.h(aux1)
        qc.measure(aux1, 0)

        qc.h(aux2)
        for i in range(self.latent_dim):
            qc.cswap(aux2, self.latent_dim + i, 2 * self.latent_dim + i)
        qc.h(aux2)
        qc.measure(aux2, 1)

        # Run the circuit
        results = self.sampler.run([qc], shots=1024).result()
        counts = results.get_counts()
        prob_same = counts.get("00", 0) / 1024.0
        return prob_same

    def train(self,
              data: np.ndarray,
              epochs: int = 50,
              learning_rate: float = 0.01,
              optimizer_cls=COBYLA) -> None:
        """
        Variational training of the encoder ansatz to minimize reconstruction error.
        Uses the COBYLA optimizer on the amplitude‑overlap loss.
        """
        # Build a SamplerQNN that outputs the reconstructed state
        qr = QuantumRegister(self.num_features, name="q")
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)
        self._build_encoder(qc, list(range(self.num_features)))
        self._build_decoder(qc, list(range(self.num_features)))

        qnn = SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=lambda x: x,
            output_shape=self.num_features,
            sampler=self.sampler,
        )

        opt = optimizer_cls()
        for _ in range(epochs):
            loss = 0.0
            for x in data:
                # Forward pass
                preds = qnn.forward(torch.as_tensor(x, dtype=torch.float32))
                loss += torch.mean((preds - torch.as_tensor(x)) ** 2).item()
            loss /= len(data)
            opt.optimize(lambda p: loss, len(qnn.weight_params))
