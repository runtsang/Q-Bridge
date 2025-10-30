"""
Quantum implementation of a hybrid estimator that mirrors the classical
HybridEstimatorNet.  The circuit encodes the input features using a
RealAmplitudes ansatz, prepares a latent register that represents the
classical latent vector, and uses a swap‑test to compute the overlap
between the two states.  The overlap is interpreted directly as the
regression output.

The design is intentionally lightweight: the circuit contains only
real‑parameter rotations, a few controlled‑swap gates, and a single
measurement.  It is compatible with Qiskit’s StatevectorSampler
and can be trained with the COBYLA optimizer or any gradient‑free
method.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def EstimatorAutoencoderQNN(
    input_dim: int,
    latent_dim: int,
    reps: int = 3,
    seed: int | None = 42,
) -> SamplerQNN:
    """
    Build a quantum neural network that implements the swap‑test
    between an input state and a latent state.

    Parameters
    ----------
    input_dim : int
        Number of classical input features.
    latent_dim : int
        Size of the latent register (matches the classical latent vector).
    reps : int
        Number of repetitions for the RealAmplitudes ansatz.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    qnn : SamplerQNN
        A Qiskit neural network object ready for training.
    """
    algorithm_globals.random_seed = seed

    # ------------------ Input ansatz ------------------ #
    input_ansatz = RealAmplitudes(input_dim, reps=reps)

    # ------------------ Latent ansatz ------------------ #
    latent_ansatz = RealAmplitudes(latent_dim, reps=reps)

    # ------------------ Build the full circuit ------------------ #
    total_qubits = input_dim + latent_dim + 1  # +1 ancillary for swap test
    qc = QuantumCircuit(total_qubits)
    # Encode input
    qc.compose(input_ansatz, range(input_dim), inplace=True)
    # Encode latent
    qc.compose(latent_ansatz, range(input_dim, input_dim + latent_dim), inplace=True)

    # Ancillary qubit for swap test
    ancilla = total_qubits - 1
    qc.h(ancilla)
    # Controlled swaps between input and latent qubits
    for i in range(latent_dim):
        qc.cswap(ancilla, i, input_dim + i)
    qc.h(ancilla)
    qc.measure(ancilla, 0)

    # ------------------ Sampler and interpret ------------------ #
    sampler = Sampler()
    # Interpret the measurement outcome as a scalar overlap
    def interpret(x: Statevector) -> float:
        # The probability of measuring |0⟩ on the ancilla is (1 + |⟨ψ|φ⟩|²)/2
        prob0 = x.probabilities_dict().get("0" * total_qubits, 0.0)
        overlap = 2 * prob0 - 1  # invert the relation
        return float(overlap)

    qnn = SamplerQNN(
        circuit=qc,
        input_params=input_ansatz.parameters,
        weight_params=latent_ansatz.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn

__all__ = ["EstimatorAutoencoderQNN"]
