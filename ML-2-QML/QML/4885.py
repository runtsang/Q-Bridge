"""Quantum counterpart of :class:`SharedClassName`.

The circuit first applies a 2×2 quanvolution to embed the classical image
into a 4‑qubit state, then runs a variational autoencoder with a swap test
to produce a latent representation. The output is passed to a
`SamplerQNN` to obtain a probability distribution.

The module mimics the API of the classical implementation so that it can
serve as a drop‑in replacement in experiments that require a quantum
sampler.
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit.random import random_circuit

# Quantum convolution helper (quanvolution)
class QuanvCircuit:
    """2×2 quanvolution that maps a 4‑element pixel block to a 4‑qubit state."""
    def __init__(self, backend, shots, threshold):
        self.n_qubits = 4
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Return the average probability of measuring |1> over all qubits."""
        # data shape (4,)
        data = data.reshape(1, self.n_qubits)
        binds = [{self.theta[i]: np.pi if val > self.threshold else 0} for i, val in enumerate(data[0])]
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=[binds])
        counts = job.result().get_counts(self._circuit)
        ones = sum(int(bit) for key in counts for bit in key) * counts[key]
        return ones / (self.shots * self.n_qubits)

# Variational auto‑encoder circuit (swap test + RealAmplitudes)
def auto_encoder_circuit(num_latent, num_trash):
    qr = qiskit.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = qiskit.ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

class SharedClassName:
    """Quantum version of the hybrid sampler."""
    def __init__(self, conv_backend=None, conv_shots=100, conv_threshold=127,
                 num_latent=3, num_trash=2):
        self.backend = conv_backend or AerSimulator()
        self.conv_filter = QuanvCircuit(self.backend, conv_shots, conv_threshold)
        self.autoencoder_circuit = auto_encoder_circuit(num_latent, num_trash)
        self.sampler = SamplerQNN(
            circuit=self.autoencoder_circuit,
            input_params=[],
            weight_params=self.autoencoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=StatevectorSampler(),
        )

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Execute the full pipeline on a 2×2 pixel block.

        Parameters
        ----------
        image : np.ndarray
            2×2 array of pixel intensities.

        Returns
        -------
        np.ndarray
            Probability vector of shape (2,).
        """
        # Step 1: quanvolution
        conv_out = self.conv_filter.run(image)
        # The convolution output can be used as a feature or parameter; here it is
        # ignored in the sampler evaluation for simplicity.
        # Step 2: evaluate sampler on the flattened image
        probs = self.sampler.evaluate(
            [image.ravel()],  # sample data (flattened)
        )
        return probs[0]

__all__ = ["SharedClassName"]
