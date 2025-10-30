from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

class QuantumAutoencoder:
    """
    Variational quantum autoencoder returning a classical latent vector.
    The circuit implements a swap‑test style encoder with optional trash qubits.
    """
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.sampler = Sampler()
        self.num_qubits = latent_dim + 2 * trash_dim + 1
        self.ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)

    def _build_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational block on latent + trash qubits
        qc.append(self.ansatz, range(self.latent_dim + self.trash_dim))

        qc.barrier()
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)

        # Swap‑test between latent and trash qubits
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)

        qc.h(aux)
        qc.measure(aux, cr[0])

        # Bind parameters
        for idx, val in enumerate(params):
            qc.set_parameter(idx, val)

        return qc

    def encode(self, input_vector: np.ndarray) -> np.ndarray:
        """Run the variational circuit and return the real part of the statevector."""
        qc = self._build_circuit(input_vector)
        result = self.sampler.run(qc, shots=1).result()
        state = Statevector(result.get_statevector())
        return state.data.real[: self.latent_dim]

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Placeholder for a classical decoder; user can attach a neural net."""
        return latent
