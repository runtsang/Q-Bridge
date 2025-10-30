"""Hybrid quantum autoencoder with integrated self‑attention."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """Variational autoencoder that uses a quantum self‑attention sub‑circuit."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        device: str = "qasm_simulator",
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.backend = Aer.get_backend(device)

        # Build the overall circuit
        self.circuit = self._build_circuit()

        # SamplerQNN to evaluate the circuit
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.circuit.parameters,
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )

    def _self_attention_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Quantum self‑attention block."""
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(num_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        # Random rotations
        for i in range(num_qubits):
            qc.rx(np.random.rand(), i)
            qc.ry(np.random.rand(), i)
            qc.rz(np.random.rand(), i)
        # Controlled‑RX entangling
        for i in range(num_qubits - 1):
            qc.crx(np.random.rand(), i, i + 1)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Compose the full variational autoencoder circuit."""
        # Feature embedding
        feature_circ = RawFeatureVector(self.input_dim, self.input_dim)

        # Self‑attention sub‑circuit
        attn_circ = self._self_attention_circuit(self.input_dim)

        # Variational encoding
        var_circ = RealAmplitudes(
            self.latent_dim + self.num_trash + 1, reps=5
        )

        # Swap test (auxiliary qubit)
        qc = QuantumCircuit(
            self.latent_dim + 2 * self.num_trash + 1, 1
        )
        qc.compose(var_circ, range(0, self.latent_dim + self.num_trash), inplace=True)
        qc.barrier()

        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, 0)

        # Combine all parts
        total_circ = QuantumCircuit()
        total_circ.compose(feature_circ, inplace=True)
        total_circ.compose(attn_circ, inplace=True)
        total_circ.compose(qc, inplace=True)

        return total_circ

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the autoencoder on a single data point."""
        result = self.qnn.forward(x)
        return result

def HybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int = 3,
    num_trash: int = 2,
    device: str = "qasm_simulator",
) -> HybridAutoencoder:
    """Convenience constructor mirroring the classical factory."""
    return HybridAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_trash=num_trash,
        device=device,
    )

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory"]
