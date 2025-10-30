"""Quantum estimator that embeds an autoencoder‑style swap‑test circuit and uses a sampler QNN."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from typing import Sequence, List

class EstimatorQNNGen392:
    """
    Quantum estimator that builds a swap‑test autoencoder circuit using RealAmplitudes
    and evaluates it with a StatevectorSampler‑based SamplerQNN.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x[0].real,
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash qubits via RealAmplitudes
        qc.compose(
            RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps),
            range(0, self.num_latent + self.num_trash),
            inplace=True,
        )
        qc.barrier()

        # Swap‑test with trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc

    def evaluate(self, weight_sets: Sequence[Sequence[float]]) -> List[float]:
        """
        Evaluate the quantum neural network for each weight set.
        """
        return self.qnn.evaluate([], weight_sets)

def EstimatorQNN(*args, **kwargs):
    """Convenience wrapper matching the original anchor signature."""
    return EstimatorQNNGen392(*args, **kwargs)

__all__ = ["EstimatorQNN", "EstimatorQNNGen392"]
