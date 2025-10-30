"""Quantum hybrid autoencoder with a self‑attention block.

The circuit first encodes the input data with a RealAmplitudes ansatz,
then applies a quantum self‑attention block on the latent qubits, and
finally performs a swap‑test for decoding.  The resulting SamplerQNN can
be trained with a classical optimizer and serves as a direct quantum
counterpart to the classical HybridAttentionAutoencoder above.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# Quantum self‑attention helper (mirroring the original)
def _build_quantum_self_attention_circuit(num_qubits: int) -> QuantumCircuit:
    """Return a circuit implementing a simple quantum self‑attention block."""
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Rotation parameters
    rotation_params = [Parameter(f"rx_{i}") for i in range(num_qubits)]
    # Entanglement parameters
    entangle_params = [Parameter(f"crx_{i}") for i in range(num_qubits - 1)]

    for i in range(num_qubits):
        circuit.rx(rotation_params[i], i)
        circuit.ry(rotation_params[i], i)
        circuit.rz(rotation_params[i], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit

def QuantumHybridAttentionAutoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    reps: int = 5,
) -> SamplerQNN:
    """Factory that builds a hybrid quantum autoencoder with self‑attention."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # Encoder ansatz
    total_qubits = num_latent + num_trash
    ansatz = RealAmplitudes(total_qubits, reps=reps)

    # Attention block on the latent qubits
    attention_circuit = _build_quantum_self_attention_circuit(num_latent)

    # Full autoencoder circuit
    qr = QuantumRegister(total_qubits + 1, "q")  # auxiliary qubit for swap test
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode
    circuit.compose(ansatz, range(total_qubits), inplace=True)

    # Attention on latent qubits
    circuit.compose(attention_circuit, range(num_latent), inplace=True)

    # Swap‑test decoding
    circuit.h(total_qubits)  # auxiliary qubit
    for i in range(num_trash):
        circuit.cswap(total_qubits, num_latent + i, num_latent + num_trash + i)
    circuit.h(total_qubits)
    circuit.measure(total_qubits, cr[0])

    # Weight parameters: ansatz + attention
    weight_params = list(ansatz.parameters) + list(attention_circuit.parameters)

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=weight_params,
        interpret=lambda x: x,  # identity
        output_shape=2,
        sampler=sampler,
    )
    return qnn

class HybridAttentionAutoencoder:
    """Convenient wrapper exposing a run method around the SamplerQNN."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
    ) -> None:
        self.qnn = QuantumHybridAttentionAutoencoder(num_latent, num_trash, reps)

    def run(self, inputs: np.ndarray, shots: int = 1024) -> dict:
        """Execute the quantum circuit and return measurement counts."""
        # The SamplerQNN expects a list of input parameters; here there are none.
        return self.qnn.forward(inputs, shots=shots)

__all__ = ["HybridAttentionAutoencoder"]
