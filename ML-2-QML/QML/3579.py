"""Quantum auto‑encoder with a quantum self‑attention sub‑circuit.

The construction follows the same factory pattern as the classical
counterpart, enabling side‑by‑side benchmarking.  The circuit uses a
RealAmplitudes encoder, a domain‑wall style attention module, and a
SamplerQNN wrapper for fast expectation evaluation.

Typical usage::
    >>> from Autoencoder__gen207 import Autoencoder
    >>> qnn = Autoencoder()
    >>> loss_history = train_autoencoder_qgen207(qnn, data, shots=1024)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Quantum self‑attention helper
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Builds a parametric attention sub‑circuit on ``n_qubits``."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Return a circuit that applies a small attention pattern."""
        qc = QuantumCircuit(self.qr, self.cr)
        # Rotate each qubit with a small 3‑parameter gate
        for i in range(self.n_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        # Pairwise controlled‑X to entangle neighboring qubits
        for i in range(self.n_qubits - 1):
            qc.crx(params[self.n_qubits + i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, params: np.ndarray, shots: int = 1024):
        qc = self._circuit(params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder
# --------------------------------------------------------------------------- #
class AutoencoderGen207:
    """Quantum auto‑encoder that embeds a self‑attention block.

    The circuit consists of:
      1. A RealAmplitudes encoder mapping the input to a latent subspace.
      2. A domain‑wall style attention sub‑circuit operating on the latent qubits.
      3. A decoder that reconstructs the input via a simple measurement.
    """

    def __init__(self, *, input_dim: int, latent_dim: int = 32,
                 attention_qubits: int = 4, shots: int = 1024) -> None:
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attention_qubits = attention_qubits
        self.shots = shots

        self.sampler = Sampler()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Build the main encoder circuit
        self.encoder = RealAmplitudes(latent_dim, reps=3)

        # Attention sub‑circuit
        self.attention = QuantumSelfAttention(attention_qubits)

        # Assemble full auto‑encoder circuit
        self.circuit = self._build_circuit()

        # SamplerQNN wrapper
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self._get_all_params(),
            interpret=lambda x: x,  # raw counts
            output_shape=2,        # binary measurement result
            sampler=self.sampler,
        )

    def _get_all_params(self) -> List[qiskit.circuit.Parameter]:
        """Collect all trainable parameters from encoder and attention."""
        return list(self.encoder.parameters()) + list(self.attention._circuit(qiskit.circuit.ParameterVector("att", len(self.attention._circuit(np.zeros(0)).parameters()))).parameters())

    def _build_circuit(self) -> QuantumCircuit:
        """Return the combined auto‑encoder circuit."""
        total_qubits = self.latent_dim + self.attention_qubits
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(total_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode
        qc.compose(self.encoder, range(self.latent_dim), inplace=True)

        # Attention block on the remaining qubits
        qc.compose(self.attention._circuit(np.zeros(len(self.attention._circuit(np.zeros(0)).parameters()))), range(self.latent_dim, total_qubits), inplace=True)

        # Simple measurement to read out the latent state
        qc.measure(qr, cr)
        return qc

    def run(self, params: np.ndarray, shots: int = None) -> dict:
        """Execute the sampler quantum circuit with the given parameters."""
        shots = shots or self.shots
        qc = self.circuit
        # Bind parameters
        param_map = {p: v for p, v in zip(self._get_all_params(), params)}
        bound_qc = qc.bind_parameters(param_map)
        job = self.backend.run(bound_qc, shots=shots)
        return job.result().get_counts(bound_qc)


def Autoencoder(*, input_dim: int, latent_dim: int = 32,
                attention_qubits: int = 4, shots: int = 1024) -> AutoencoderGen207:
    """Factory for a quantum :class:`AutoencoderGen207`."""
    return AutoencoderGen207(
        input_dim=input_dim,
        latent_dim=latent_dim,
        attention_qubits=attention_qubits,
        shots=shots,
    )


def train_autoencoder_qgen207(
    qnn: AutoencoderGen207,
    data: np.ndarray,
    *,
    iterations: int = 50,
    base_lr: float = 0.01,
) -> List[float]:
    """Train the quantum auto‑encoder using a simple COBYLA optimizer.

    The loss is computed as the mean Hamming distance between the
    measurement outcome and the original binary input.
    """
    optimizer = COBYLA(maxiter=iterations)

    # Flatten data to binary strings (simple one‑hot for demo)
    inputs = (data > data.mean()).astype(int)
    target_counts = {f"{int(b)}": 1 for b in inputs.flatten()}

    loss_history: List[float] = []

    def objective(params: np.ndarray) -> float:
        counts = qnn.run(params, shots=qnn.shots)
        # Convert counts to a probability distribution over the input bits
        probs = np.array([counts.get(f"{i:0{qnn.latent_dim}b}", 0) for i in range(2**qnn.latent_dim)]) / qnn.shots
        # Hamming distance loss
        loss = np.mean([np.sum(np.abs(int(b) - probs[i])) for i, b in enumerate(inputs)])
        loss_history.append(loss)
        return loss

    # Initial random parameters
    init_params = np.random.uniform(0, 2 * np.pi, len(qnn._get_all_params()))
    optimizer.optimize(init_params, objective)
    return loss_history


__all__ = [
    "AutoencoderGen207",
    "Autoencoder",
    "train_autoencoder_qgen207",
]
