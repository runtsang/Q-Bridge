"""Hybrid classical/quantum estimator – quantum side.

The module defines a Qiskit‑based variational circuit that receives the
classical latent vector produced by the auto‑encoder.  It employs a
domain‑wall inspired ansatz built on `RealAmplitudes` and a `SamplerQNN`
for efficient state‑vector sampling.  The circuit outputs a single
regression value via a simple interpret function.

The design mirrors the quantum auto‑encoder seed while adding a linear
read‑out that is compatible with the classical estimator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Insert X gates on a contiguous block to emulate a domain wall."""
    for i in range(start, end):
        circuit.x(i)
    return circuit

def _build_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct the core variational block that encodes latent data."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode latent data into first qubits via RealAmplitudes
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    # Domain wall on the remaining trash qubits
    qc = _domain_wall(qc, num_latent, num_latent + num_trash)

    # Swap‑test style read‑out on auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

class QuantumHybridEstimator:
    """
    A quantum neural network that maps a classical latent vector to a
    regression value.

    Parameters
    ----------
    latent_dim : int
        Dimension of the classical latent vector produced by the auto‑encoder.
    num_trash : int
        Number of auxiliary qubits used for the domain‑wall construction.
    """

    def __init__(self, latent_dim: int, num_trash: int = 2) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.circuit = _build_encoder_circuit(latent_dim, num_trash)
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],  # no trainable input parameters
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=1,
            sampler=self.sampler,
        )

    def _interpret(self, result: np.ndarray) -> float:
        """
        Convert sampler output into a scalar regression value.

        The sampler returns a probability distribution over the
        measurement outcomes of the auxiliary qubit.  We interpret the
        probability of measuring |1⟩ as the predicted value.
        """
        p_one = result[1]
        return float(p_one)

    def predict(self, latent: np.ndarray) -> float:
        """
        Evaluate the QNN on a single latent vector.

        Parameters
        ----------
        latent : np.ndarray
            Shape ``(latent_dim,)``.  The vector is first expanded to the
            required number of qubits by padding with zeros.

        Returns
        -------
        float
            Regression prediction.
        """
        # Pad latent vector to match the first `latent_dim` qubits
        padded = np.zeros(self.circuit.num_qubits)
        padded[:self.latent_dim] = latent
        # The QNN expects a dict mapping parameters to values; here we
        # provide the raw state vector as the input.
        res = self.qnn(padded)
        return self._interpret(res)

    def __call__(self, latent: np.ndarray) -> float:
        return self.predict(latent)

__all__ = ["QuantumHybridEstimator"]
