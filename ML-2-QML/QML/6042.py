"""Hybrid quantum autoencoder with a quantum self‑attention sub‑circuit."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Quantum self‑attention block
class QuantumSelfAttention:
    """A parameterized quantum circuit that mimics a self‑attention operation."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circ = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)

# Hybrid autoencoder circuit
def HybridQuantumAutoencoder(num_latent: int = 3, num_trash: int = 2, n_attention: int = 4):
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def encoder_circuit(num_qubits: int) -> QuantumCircuit:
        """Encode the input into the first `num_qubits` qubits."""
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        qc.append(RealAmplitudes(num_qubits, reps=3), qr)
        return qc

    def attention_block(num_qubits: int) -> QuantumCircuit:
        """Quantum self‑attention sub‑circuit."""
        qa = QuantumSelfAttention(n_qubits=num_qubits)
        # Dummy parameters – in practice these would be learned
        rotation_params = np.random.rand(3 * num_qubits)
        entangle_params = np.random.rand(num_qubits - 1)
        return qa._build_circuit(rotation_params, entangle_params)

    def decoder_circuit(num_qubits: int) -> QuantumCircuit:
        """Decode the latent back to the original dimensionality."""
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        qc.append(RealAmplitudes(num_qubits, reps=3), qr)
        return qc

    # Full circuit composition
    total_qubits = num_latent + 2 * num_trash + 1
    qc = QuantumCircuit(total_qubits, 1)

    # Encode
    qc.compose(encoder_circuit(num_latent), range(0, num_latent), inplace=True)

    # Add quantum self‑attention
    qc.compose(attention_block(num_latent), range(0, num_latent), inplace=True)

    # Swap test for reconstruction
    trash_start = num_latent
    for i in range(num_trash):
        qc.cswap(0, num_latent + i, trash_start + i)
    qc.h(0)
    qc.measure(0, 0)

    # Decoder (optional – here we simply reuse the encoder as a mock decoder)
    qc.compose(decoder_circuit(num_latent), range(0, num_latent), inplace=True)

    # Wrap into a QNN
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,  # identity interpret
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["HybridQuantumAutoencoder"]
