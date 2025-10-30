"""
Hybrid quantum autoencoder sampler.

This module fuses the quantum autoencoder circuit from reference pair 2
with the SamplerQNN helper from reference pair 1.  It constructs a
RealAmplitudes ansatz followed by a domain‑wall and swap‑test that
produces a single classical bit measurement.  The resulting
:class:`SamplerQNN` can be used as a variational quantum circuit
whose weights are trained to minimise a reconstruction loss.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals


class HybridAutoSamplerQNN:
    """
    Quantum autoencoder sampler.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits used in the ansatz.
    num_trash : int
        Number of trash qubits that participate in the domain‑wall and
        swap‑test operations.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        algorithm_globals.random_seed = 42
        self.sampler = Sampler()
        self.circuit = self._build_circuit(num_latent, num_trash)
        # The circuit has no input parameters; weights are the ansatz
        # parameters.  We expose the underlying SamplerQNN.
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        """Construct the quantum autoencoder circuit."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz: RealAmplitudes over the latent + trash subspace
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

        qc.barrier()

        # Domain‑wall: flip the first `num_trash` qubits
        for i in range(num_trash):
            qc.x(num_latent + i)

        # Swap‑test auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)

        # Measure auxiliary qubit into the classical register
        qc.measure(aux, cr[0])
        return qc

    def __call__(self, *weights: np.ndarray | list[float]) -> SamplerQNN:
        """
        Update the weights of the underlying SamplerQNN.

        Parameters
        ----------
        weights
            Iterable of weight values matching the circuit parameters.

        Returns
        -------
        SamplerQNN
            The wrapped quantum neural network ready for evaluation.
        """
        if weights:
            self.qnn.set_weights(weights)
        return self.qnn


def HybridAutoSamplerQNNFactory(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """
    Factory that returns a ready‑to‑use :class:`SamplerQNN` instance.
    """
    return HybridAutoSamplerQNN(num_latent, num_trash).qnn


__all__ = ["HybridAutoSamplerQNN", "HybridAutoSamplerQNNFactory"]
