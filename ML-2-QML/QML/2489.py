import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridQuantumAutoencoder:
    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 3):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(self.num_latent,),
            sampler=self.sampler
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature encoding with RealAmplitudes on latent+trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, list(range(self.num_latent + self.num_trash)), inplace=True)

        # Swap test to compare trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the quantum autoencoder.
        x should be a 1‑D array of length equal to the number of circuit parameters.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        preds = self.qnn.predict(x)
        return preds

def HybridQuantumAutoencoderFactory(num_latent: int = 3, num_trash: int = 2, reps: int = 3) -> HybridQuantumAutoencoder:
    return HybridQuantumAutoencoder(num_latent=num_latent, num_trash=num_trash, reps=reps)

__all__ = ["HybridQuantumAutoencoder", "HybridQuantumAutoencoderFactory"]
