"""HybridAutoencoder – quantum implementation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import Parameter

class HybridAutoencoder:
    """
    Quantum auto‑encoder built with a RealAmplitudes ansatz, a swap‑test
    for latent extraction, and a SamplerQNN interface.  The circuit
    accepts no explicit input parameters – in a real use‑case the data
    would be encoded via a feature map before the ansatz.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the variational auto‑encoder circuit."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        qc.barrier()

        # Swap‑test with an auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def _interpret(self, samples: np.ndarray) -> np.ndarray:
        """Map raw sampler samples to a 2‑dimensional output."""
        # samples shape: (n_samples, 1)
        p1 = np.mean(samples, axis=0)[0]
        p0 = 1.0 - p1
        return np.array([p0, p1])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the quantum auto‑encoder on the provided inputs.
        In this toy example inputs are ignored; they would be encoded into
        the initial state in a real implementation.
        Returns a 2‑dimensional array per input.
        """
        # Run the same circuit for each input; in practice the circuit would
        # depend on the encoded data.
        return np.array([self.qnn([{}]) for _ in inputs])

__all__ = ["HybridAutoencoder"]
