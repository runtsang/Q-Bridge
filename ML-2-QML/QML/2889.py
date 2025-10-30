"""AutoencoderGen153 – Quantum implementation using Qiskit and SamplerQNN.

The circuit mirrors the classical autoencoder structure:
an encoder ansatz, a domain‑wall injection, and a swap‑test based
decoder.  The QNN is wrapped in a SamplerQNN for efficient
state‑vector evaluation, and a COBYLA optimizer is provided for
variational training.  The design is intentionally modular so that
the quantum block can be swapped with a classical one for benchmarking.
"""

from __future__ import annotations

import numpy as np
import time
from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Insert X gates on qubits [start, end) to create a domain wall."""
    for i in range(start, end):
        circuit.x(i)
    return circuit


# ----------------------------------------------------------------------
# Core quantum autoencoder
# ----------------------------------------------------------------------
class AutoencoderGen153:
    """Quantum autoencoder using a RealAmplitudes ansatz with a domain‑wall and swap‑test."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        device: str | None = None,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = StatevectorSampler(device=device)
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(maxiter=200)

    # ------------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full quantum circuit."""
        total_qubits = self.num_latent + 2 * self.num_trash + 1  # +1 auxiliary
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder ansatz
        qc.append(RealAmplitudes(total_qubits, reps=self.reps), range(total_qubits))

        qc.barrier()

        # Auxiliary qubit for swap‑test
        aux = total_qubits - 1
        qc.h(aux)

        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)

        qc.h(aux)
        qc.measure(aux, cr[0])

        # Domain wall injection on a subset of qubits (e.g., the first 5)
        if total_qubits >= 5:
            domain_circ = _domain_wall(QuantumCircuit(total_qubits), 0, 5)
            qc = qc.compose(domain_circ, range(total_qubits))

        return qc

    # ------------------------------------------------------------------
    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 10,
        verbose: bool = False,
    ) -> List[float]:
        """Variational training using COBYLA on the sampler QNN."""
        history: List[float] = []
        weights = np.random.uniform(0, 2 * np.pi, size=len(self.circuit.parameters))

        for epoch in range(epochs):
            def loss_fn(params: np.ndarray) -> float:
                self.qnn.set_weights(params)
                # Expectation value of the auxiliary qubit measurement
                probs = self.qnn.predict(np.zeros((1, 0)))  # no inputs
                # Target: minimize probability of measuring |1> (i.e., fidelity)
                return probs[0, 1]

            weights = self.optimizer.optimize(loss_fn, weights)
            loss = loss_fn(weights)
            history.append(loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")
        return history

    # ------------------------------------------------------------------
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum latent space (placeholder)."""
        # For demonstration, we simply return the input reshaped to the
        # expected number of qubits.  In practice, a feature‑map would be applied.
        return inputs.reshape((-1, self.num_latent))

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode quantum latents back to classical space (placeholder)."""
        return latents.reshape((-1, self.num_latent))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Full autoencoding pass."""
        latent = self.encode(inputs)
        decoded = self.decode(latent)
        return decoded


__all__ = ["AutoencoderGen153"]
