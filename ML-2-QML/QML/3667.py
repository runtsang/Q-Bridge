"""Quantum Quanvolution Autoencoder using Qiskit.

The circuit first encodes the image patch via a domain‑wall pattern,
then applies a RealAmplitudes ansatz, followed by a swap test between
latent and trash qubits.  The resulting expectation values are fed
into a classical decoder.
"""

import numpy as np
import qiskit as qs
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumQuanvolutionAutoencoder:
    """Hybrid quantum–classical autoencoder.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits that encode the compressed representation.
    num_trash : int
        Number of auxiliary qubits used in the swap test.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=np.squeeze,
            output_shape=(self.num_latent,),
            sampler=self.sampler,
        )

    def _domain_wall(self, qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """Insert a domain wall (X gates) on qubits [start, end)."""
        for i in range(start, end):
            qc.x(i)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Domain wall pattern on trash qubits
        qc = self._domain_wall(qc, self.num_latent, self.num_latent + self.num_trash)

        # Ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qc.compose(ansatz, range(self.num_latent + self.num_trash), inplace=True)

        # Swap test between latent and trash
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the compressed representation of input data.

        Parameters
        ----------
        inputs : array of shape (batch, num_latent + num_trash)
            Classical feature vectors that will be encoded as circuit parameters.
        """
        # Map classical inputs to circuit parameters
        circ = self.circuit.assign_parameters(inputs, inplace=False)
        result = self.sampler.run(circ, shots=1024).result()
        probs = result.get_counts(circ)
        # Convert to expectation values
        exp_vals = np.array(
            [2 * probs.get(bin(i)[2:].zfill(1), 0) / 1024 - 1 for i in range(self.num_latent)]
        )
        return exp_vals

__all__ = ["QuantumQuanvolutionAutoencoder"]
