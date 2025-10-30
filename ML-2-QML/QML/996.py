import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class QuantumAutoencoder:
    """Hybrid variational autoencoder that embeds classical data into a unitary and reconstructs via swap‑test."""
    def __init__(self, num_latent: int, num_trash: int, reps: int = 3) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a parameterised circuit with an embedding layer and a swap‑test."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Feature embedding
        embed = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(embed, range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()
        # Swap‑test block
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def get_qnn(self, sampler: StatevectorSampler | None = None) -> SamplerQNN:
        """Return a SamplerQNN that evaluates the swap‑test probability."""
        if sampler is None:
            sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(2,),
            sampler=sampler,
        )

def quantum_autoencoder(device_seed: int = 42) -> QuantumAutoencoder:
    """Convenience wrapper that returns a ready‑to‑train QNN."""
    algorithm_globals.random_seed = device_seed
    return QuantumAutoencoder(num_latent=3, num_trash=2, reps=4)

__all__ = ["QuantumAutoencoder", "quantum_autoencoder"]
