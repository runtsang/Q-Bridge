"""Quantum‑aware transformer and autoencoder using Qiskit.

The module implements a quantum autoencoder with a swap‑test based latent extraction,
and a quantum transformer block that applies a parameterised attention and feed‑forward
circuit.  The public class `HybridTransformerAutoencoder` mirrors the classical API
but builds the entire model as a Qiskit circuit wrapped in a `SamplerQNN`.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


# ---------- Quantum Autoencoder ----------
def quantum_autoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
    """
    Builds a swap‑test based quantum autoencoder.
    `num_latent` qubits carry the latent representation.
    `num_trash` qubits are used as ancillae.
    Returns a SamplerQNN ready for training.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz for encoding
    circuit.compose(RealAmplitudes(num_latent + num_trash, reps=4), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    # Sampler
    sampler = AerSimulator()
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )


# ---------- Quantum Transformer Block ----------
def quantum_transformer_block(num_qubits: int, num_heads: int, depth: int = 1) -> QuantumCircuit:
    """
    Builds a single transformer‑style block with quantum attention and feed‑forward.
    `num_qubits` is the number of qubits representing the token embeddings.
    """
    qc = QuantumCircuit(num_qubits)

    # Attention ansatz
    qc.compose(RealAmplitudes(num_qubits, reps=3), inplace=True)
    qc.barrier()

    # Simple feed‑forward using RY gates
    for q in range(num_qubits):
        qc.ry(0.5, q)
    qc.barrier()

    # Depth repetitions
    for _ in range(depth - 1):
        qc.compose(RealAmplitudes(num_qubits, reps=3), inplace=True)
        for q in range(num_qubits):
            qc.ry(0.5, q)

    return qc


# ---------- Hybrid Transformer Autoencoder ----------
class HybridTransformerAutoencoder:
    """
    Quantum implementation that mirrors the classical `HybridTransformerAutoencoder`.
    The model is represented as a single variational circuit composed of
    transformer blocks followed by a quantum autoencoder.  The circuit is exposed
    as a `SamplerQNN` for easy integration with Qiskit Machine Learning.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        num_latent: int,
        num_trash: int,
        depth: int = 1,
        seed: int = 42,
    ) -> None:
        algorithm_globals.random_seed = seed
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.depth = depth

        # Build the parameterised circuit
        self.circuit = self._build_circuit()

        # Sampler and QNN
        self.sampler = AerSimulator()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=self.num_classes if self.num_classes > 2 else 1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Compose transformer blocks and the autoencoder."""
        qc = QuantumCircuit(self.embed_dim)

        # Token embedding is simulated by a classical lookup and then a
        # RealAmplitudes ansatz to encode the token into qubits.
        qc.compose(RealAmplitudes(self.embed_dim, reps=2), inplace=True)

        # Transformer blocks
        for _ in range(self.num_blocks):
            block = quantum_transformer_block(self.embed_dim, self.num_heads, self.depth)
            qc.compose(block, inplace=True)

        # Autoencoder
        ae = quantum_autoencoder(self.num_latent, self.num_trash)
        qc.compose(ae.circuit, inplace=True)

        return qc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the variational circuit.
        `x` is expected to be a token sequence encoded as integers.
        For demonstration, the input is ignored and a single execution is performed.
        """
        result = self.sampler.run(self.circuit).result()
        counts = result.get_counts()
        return np.array(list(counts.values()))


__all__ = ["HybridTransformerAutoencoder"]
