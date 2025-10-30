"""Quantum autoencoder with a self‑attention block.

The module mirrors the classical Autoencoder API while using a variational
quantum circuit.  The circuit consists of a RealAmplitudes ansatz for the
latent qubits followed by a self‑attention block that entangles the latent
and trash qubits.  The final state is sampled with Qiskit's StatevectorSampler
and wrapped in a SamplerQNN.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
backend = qiskit.Aer.get_backend("qasm_simulator")


def _build_self_attention_circuit(
    qr: QuantumRegister, cr: ClassicalRegister, rotation_params: np.ndarray, entangle_params: np.ndarray
) -> QuantumCircuit:
    """Construct a simple self‑attention style block."""
    circuit = QuantumCircuit(qr, cr)
    for i in range(qr.size):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)
    for i in range(qr.size - 1):
        circuit.crx(entangle_params[i], i, i + 1)
    circuit.measure(qr, cr)
    return circuit


class HybridAutoencoder:
    """Variational quantum autoencoder with an embedded self‑attention block."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Total qubits: latent + trash (one per hidden dim)
        num_trash = len(hidden_dims)
        num_qubits = latent_dim + 2 * num_trash + 1  # +1 for auxiliary

        self.qr = QuantumRegister(num_qubits, "q")
        self.cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Encode: RealAmplitudes ansatz on latent + trash qubits
        ansatz = RealAmplitudes(latent_dim + num_trash, reps=5)
        self.circuit.compose(ansatz, range(0, latent_dim + num_trash), inplace=True)

        # Self‑attention block on remaining qubits
        rotation_params = np.random.uniform(0, 2 * np.pi, 3 * num_qubits)
        entangle_params = np.random.uniform(0, 2 * np.pi, num_qubits - 1)
        self.circuit.compose(
            _build_self_attention_circuit(self.qr, self.cr, rotation_params, entangle_params),
            inplace=True,
        )

        # Swap test with auxiliary qubit
        aux = num_qubits - 1
        self.circuit.h(aux)
        for i in range(num_trash):
            self.circuit.cswap(aux, latent_dim + i, latent_dim + num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, self.cr[0])

        # Build the QNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=ansatz.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=StatevectorSampler(),
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum circuit and return the measurement probabilities."""
        return self.qnn.run(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Placeholder – quantum autoencoders typically reconstruct via post‑processing."""
        return latents

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Factory that returns a configured :class:`HybridAutoencoder`."""
    return HybridAutoencoder(input_dim, latent_dim, hidden_dims, dropout)


__all__ = ["Autoencoder", "HybridAutoencoder"]
