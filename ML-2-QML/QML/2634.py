"""
QuantumQuanvolutionAutoencoder: A hybrid quantum autoencoder that
first maps image patches into a quantum state via a RealAmplitudes
ansatz, then compresses the state using a swap‑test based autoencoder
circuit.  The output is a probability vector that can be interpreted
as a compressed representation.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.circuit.library import RawFeatureVector


class QuantumQuanvolutionAutoencoder:
    """
    Parameters
    ----------
    num_latent : int
        Number of qubits that encode the compressed representation.
    num_trash : int
        Number of ancillary qubits used in the swap‑test.
    reps : int
        Number of repetitions of the RealAmplitudes ansatz.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()

        # Wrap in a SamplerQNN for easy evaluation
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """
        Constructs a circuit that:
        1. Encodes a 4‑pixel patch into a 4‑qubit state via a RealAmplitudes ansatz.
        2. Applies a swap‑test between latent and trash qubits.
        3. Measures the ancilla to obtain a compressed value.
        """
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # 1. Feature encoding
        ansatz = RealAmplitudes(num_qubits=self.num_latent + self.num_trash, reps=self.reps)
        qc.append(ansatz, range(0, self.num_latent + self.num_trash))

        # 2. Swap‑test
        qc.barrier()
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        # 3. Measurement
        qc.measure(aux, cr[0])
        return qc

    def encode(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a 2×2 pixel patch (flattened to length 4) into the quantum
        compressed representation.

        Parameters
        ----------
        patch : np.ndarray
            Array of shape (4,) with pixel intensities in [0, 1].

        Returns
        -------
        np.ndarray
            Probability distribution over the ancilla measurement outcome.
        """
        # Prepare feature vector circuit
        feature_circuit = RawFeatureVector(patch, qubits=range(self.num_latent + self.num_trash))
        combined = feature_circuit.compose(self.circuit, inplace=False)
        result = self.sampler.run(combined, shots=1024).result()
        counts = result.get_counts()
        probs = np.array([counts.get("0", 0), counts.get("1", 0)]) / 1024
        return probs

    def decode(self, compressed: np.ndarray) -> np.ndarray:
        """
        In this toy example the decoder simply maps the compressed
        probability back to a 4‑pixel vector via a linear transform.
        """
        # Simple linear mapping (placeholder)
        return compressed @ np.array([[0.25, 0.25, 0.25, 0.25],
                                      [0.25, 0.25, 0.25, 0.25]])

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Full forward pass over a 28×28 image: extract 2×2 patches,
        encode each with the quantum circuit, and flatten the results.
        """
        patches = image.reshape(14, 14, 2, 2).reshape(-1, 4)
        encoded = np.array([self.encode(p) for p in patches])
        return encoded.flatten()

__all__ = ["QuantumQuanvolutionAutoencoder"]
