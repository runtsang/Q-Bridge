"""
Pure quantum autoencoder that implements the same architecture as the hybrid
model but without the classical MLP.  The circuit consists of a self‑attention
block followed by a RealAmplitudes ansatz and a domain‑wall swap‑test.
The output is produced by a SamplerQNN.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN


def _build_qiskit_self_attention(
    num_qubits: int,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> QuantumCircuit:
    """Quantum self‑attention circuit used inside the auto‑encoder."""
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    for i in range(num_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit


class Autoencoder:
    """
    Quantum autoencoder that mirrors the structure of the hybrid model.
    It can be instantiated with a desired latent dimensionality.
    """

    def __init__(self, latent_dim: int, num_trash: int = 2):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.backend = Aer.get_backend("aer_simulator_statevector")
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(1,),
            sampler=self.backend,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # --- Self‑attention block ------------------------------------
        sa_params = np.random.rand(self.latent_dim * 3)
        ent_params = np.random.rand(self.latent_dim - 1)
        sa_circuit = _build_qiskit_self_attention(
            self.latent_dim, sa_params, ent_params
        )
        circuit.compose(sa_circuit, inplace=True)

        # --- Auto‑encoder ansatz ------------------------------------
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)

        # --- Domain‑wall swap‑test -----------------------------------
        circuit.barrier()
        aux = self.latent_dim + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        circuit.h(aux)

        circuit.measure(aux, cr[0])
        return circuit

    def encode(self, *args, **kwargs):
        """
        Forward pass: returns the sampled output of the quantum circuit.
        """
        return self.qnn.predict([])

    def decode(self, *args, **kwargs):
        """
        In this pure quantum implementation the decoder is identical to the
        encoder.  The method is provided for API consistency.
        """
        return self.qnn.predict([])
