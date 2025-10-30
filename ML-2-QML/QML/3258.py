"""Hybrid quantum autoencoder for feature extraction and reconstruction.

This module implements a variational quantum circuit that mimics a classical
autoencoder.  It uses a RealAmplitudes ansatz, a swap‑test for measuring
latent similarity, and a domain‑wall pattern for injective feature mapping.
A :class:`SamplerQNN` wrapper is included for probabilistic latent sampling.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  Sampler QNN – a lightweight quantum neural network
# --------------------------------------------------------------------------- #
class SamplerQNN:
    """A parameterised quantum sampler that outputs a probability distribution
    over two basis states.  The circuit consists of two qubits, a pair of
    Ry rotations for the inputs, a CX entanglement, and a second set of Ry
    rotations for the trainable weights.
    """

    def __init__(self, input_params: np.ndarray, weight_params: np.ndarray) -> None:
        self.input_params = input_params
        self.weight_params = weight_params
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input layer
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        # Weight layer
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        return qc

    def sample(self, shots: int = 1024) -> np.ndarray:
        """Return a probability distribution over the computational basis."""
        probs = self.sampler.run(self.circuit, shots=shots).probabilities_dict()
        # Map to a 2‑dimensional vector (|0>, |1>)
        return np.array([probs.get("0", 0.0), probs.get("1", 0.0)])


# --------------------------------------------------------------------------- #
#  Quantum Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderHybrid:
    """Variational quantum circuit that reconstructs data via a swap‑test
    and domain‑wall encoding.  The circuit is parametrised by a latent
    dimension and a trash qubit count.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2, seed: int = 42) -> None:
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()

    # --------------------------------------------------------------------- #
    #  Circuit construction helpers
    # --------------------------------------------------------------------- #
    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Return a RealAmplitudes ansatz with fixed repetitions."""
        return RealAmplitudes(num_qubits, reps=5)

    def _auto_encoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        """Build the core encoder/decoder circuit with a swap‑test."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode the first segment
        qc.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()

        # Swap‑test between latent and trash qubits
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _domain_wall(self, qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """Apply a domain‑wall pattern by X‑gates on a contiguous block."""
        for i in range(start, end):
            qc.x(i)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Assemble the full quantum autoencoder circuit."""
        core = self._auto_encoder_circuit(self.num_latent, self.num_trash)
        # Embed a domain‑wall on the middle block of qubits
        dw_size = self.num_trash
        core = self._domain_wall(core, self.num_latent, self.num_latent + dw_size)
        return core

    # --------------------------------------------------------------------- #
    #  Execution helpers
    # --------------------------------------------------------------------- #
    def sample_latent(self, shots: int = 1024) -> np.ndarray:
        """Sample the latent distribution using the StatevectorSampler."""
        sampler = StatevectorSampler()
        result = sampler.run(self.circuit, shots=shots)
        probs = result.probabilities_dict()
        # Return probabilities for the auxiliary qubit measurement outcomes
        return np.array([probs.get("0", 0.0), probs.get("1", 0.0)])

    def get_sampler_qnn(self, input_vals: np.ndarray, weight_vals: np.ndarray) -> SamplerQNN:
        """Instantiate a SamplerQNN with given parameters for further sampling."""
        return SamplerQNN(input_vals, weight_vals)

    def __repr__(self) -> str:
        return f"<AutoencoderHybrid latent={self.num_latent} trash={self.num_trash}>"


__all__ = ["AutoencoderHybrid", "SamplerQNN"]
