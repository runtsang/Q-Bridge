"""Quantum autoencoder used as a latent transformation layer."""
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, StatevectorSimulator
from qiskit.quantum_info import Pauli, Statevector
import numpy as np


class QuantumAutoencoder:
    """Quantum variational autoencoder that transforms a latent vector via a swap‑test circuit."""
    def __init__(self, latent_dim: int, num_trash: int = 2, reps: int = 5):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.num_qubits = latent_dim + 2 * num_trash + 1
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.simulator = StatevectorSimulator()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, name="q")
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)

        # placeholder for Ry rotations of latent qubits
        for i in range(self.latent_dim):
            qc.ry(0.0, i)

        # ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=self.reps)
        qc.append(ansatz.to_instruction(), list(range(self.latent_dim + self.num_trash)))

        # domain‑wall regularisation: X on the trash qubits
        for i in range(self.num_trash):
            qc.x(self.latent_dim + self.num_trash + i)

        # swap‑test with auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def _expectation_z(self, state: Statevector, qubit: int) -> float:
        """Expectation value of Pauli‑Z on a single qubit."""
        pauli_str = "I" * qubit + "Z" + "I" * (self.num_qubits - qubit - 1)
        pauli = Pauli(pauli_str)
        return float(state.expectation_value(pauli).real)

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        """Transform a batch of latent vectors via the quantum circuit."""
        batch_size = latent.shape[0]
        out = torch.empty(batch_size, self.latent_dim, dtype=torch.float32)
        for idx in range(batch_size):
            angles = latent[idx].detach().cpu().numpy()
            qc = self.circuit.copy()
            for i, angle in enumerate(angles):
                qc.ry(angle, i)
            # Run state‑vector simulation
            result = self.simulator.run(qc).result()
            state = Statevector(result.get_statevector())
            # Compute expectation values for each latent qubit
            for q in range(self.latent_dim):
                out[idx, q] = self._expectation_z(state, q)
        return out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Identity mapping – the quantum layer is meant for encoding only."""
        return latent


__all__ = ["QuantumAutoencoder"]
