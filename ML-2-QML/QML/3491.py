from __future__ import annotations

import numpy as np
from typing import Callable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumAutoEncoderCircuit:
    """
    Swap‑test based quantum encoder that refines a classical latent vector.
    The circuit encodes the latent into a subset of qubits, entangles it
    with auxiliary qubits, and measures a parity observable that
    is interpreted as a refined latent value.
    """
    def __init__(
        self,
        latent_dim: int,
        n_trash: int = 2,
        reps: int = 3,
        depth: int = 1,
        seed: int | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.n_trash = n_trash
        self.reps = reps
        self.depth = depth
        self.seed = seed or 42
        self.sampler = Sampler()
        self._build_circuit()

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Variational layer used for encoding."""
        return RealAmplitudes(num_qubits, reps=self.reps, seed=self.seed)

    def _build_circuit(self) -> None:
        """Build a full circuit that accepts a latent vector."""
        # Total qubits: latent + 2*trash + 1 (auxiliary)
        n = self.latent_dim + 2 * self.n_trash + 1
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encode latent into first `latent_dim` qubits
        encoder = self._ansatz(self.latent_dim)
        self.circuit.compose(encoder, range(0, self.latent_dim), inplace=True)

        # Entanglement with trash qubits
        for _ in range(self.depth):
            for i in range(self.n_trash):
                self.circuit.cnot(qr[self.latent_dim + i], qr[self.latent_dim + self.n_trash + i])

        # Swap‑test with auxiliary qubit
        aux = self.latent_dim + 2 * self.n_trash
        self.circuit.h(aux)
        for i in range(self.n_trash):
            self.circuit.cswap(aux, qr[self.latent_dim + i], qr[self.latent_dim + self.n_trash + i])
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def latent_to_params(self, latent: np.ndarray) -> np.ndarray:
        """Map a classical latent vector to circuit parameters."""
        # Simple linear mapping – can be replaced by a neural network.
        return latent / np.max(np.abs(latent) + 1e-12)

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        """Execute the circuit and return the measurement outcome as a refined latent."""
        params = self.latent_to_params(latent)
        # Broadcast params to match the number of qubits in the ansatz
        # (only latent qubits are parameterized)
        param_dict = {f"theta_{i}": params[i] for i in range(len(params))}
        job = self.sampler.run(self.circuit, param_binds=[param_dict], shots=1024)
        result = job.result()
        counts = result.get_counts()
        # Convert 0/1 measurement to -1/+1 and average
        avg = sum((int(k) * 2 - 1) * v for k, v in counts.items()) / sum(counts.values())
        # Return a 1‑D latent refinement; stack if multiple dimensions needed
        return np.array([avg])

def get_quantum_encoder(
    latent_dim: int,
    n_trash: int = 2,
    reps: int = 3,
    depth: int = 1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory that returns a callable converting a torch latent tensor into a
    quantum‑refined latent tensor using the above circuit.
    """
    import torch

    circuit = QuantumAutoEncoderCircuit(latent_dim, n_trash=n_trash, reps=reps, depth=depth)

    def q_enc(tensor: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy, run circuit, convert back
        np_latent = tensor.detach().cpu().numpy()
        refined = circuit(np_latent)
        return torch.from_numpy(refined).to(tensor.device).float()

    return q_enc

__all__ = [
    "QuantumAutoEncoderCircuit",
    "get_quantum_encoder",
]
