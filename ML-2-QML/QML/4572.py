"""Quantum hybrid autoencoder using a variational circuit with a trainable feed‑forward layer."""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder(SamplerQNN):
    """Quantum autoencoder with a RealAmplitudes ansatz, a trainable Ry feed‑forward
    layer, and a swap‑test decoder.  The class inherits from SamplerQNN and
    exposes a single callable that returns reconstructed probabilities."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
    ) -> None:
        # Prepare registers
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Variational ansatz on latent + first trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
        circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

        # Trainable feed‑forward layer (one Ry per latent qubit)
        feedforward_params = [Parameter(f"ry_{i}") for i in range(num_latent)]
        for i, param in enumerate(feedforward_params):
            circuit.ry(param, i)

        # Swap‑test decoder
        auxiliary = num_latent + 2 * num_trash
        circuit.h(auxiliary)
        for i in range(num_trash):
            circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary)
        circuit.measure(auxiliary, cr[0])

        # Combine all weight parameters
        weight_params = list(ansatz.params) + feedforward_params

        # Simple identity interpretation (outputs a 2‑dim vector)
        def identity_interpret(x):
            return x

        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=weight_params,
            interpret=identity_interpret,
            output_shape=2,
            sampler=Sampler(),
        )

__all__ = ["HybridAutoencoder"]
