"""Quantum estimator inspired by the quantum auto‑encoder example.

The module builds a variational circuit that encodes a latent space,
performs a swap‑test with auxiliary qubits, and returns the probability
of measuring the auxiliary qubit in the |1⟩ state.  The returned
SamplerQNN can be plugged into a hybrid training loop or used for
pure quantum inference.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Core circuit builder
# --------------------------------------------------------------------------- #
def _build_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Constructs the core auto‑encoder sub‑circuit used inside the estimator."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test that compares trash qubits
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

# --------------------------------------------------------------------------- #
# Estimator factory
# --------------------------------------------------------------------------- #
def EstimatorQNN(
    num_latent: int = 3,
    num_trash: int = 2,
    *,
    sampler: Sampler | None = None,
) -> SamplerQNN:
    """Return a quantum neural network that evaluates the probability of
    measuring |1⟩ on the auxiliary qubit after the swap‑test."""
    algorithm_globals.random_seed = 42
    if sampler is None:
        sampler = Sampler()

    qc = _build_autoencoder_circuit(num_latent, num_trash)

    # Interpret function that extracts the probability of measuring 1
    def interpret(x: np.ndarray) -> np.ndarray:
        # x has shape (samples, 1) – the measurement outcomes
        return x[:, 0].astype(np.float64)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn

__all__ = ["EstimatorQNN"]
