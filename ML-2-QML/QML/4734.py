"""Quantum autoencoder implemented with Qiskit and a variational ansatz."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def HybridAutoencoder(
    input_dim: int,
    latent_dim: int = 3,
    reps: int = 3,
) -> SamplerQNN:
    """
    Construct a quantum autoencoder circuit that encodes the input into quantum
    amplitudes, applies a variational ansatz, and uses a swap-test to
    compute fidelity between the encoded input and a latent representation.
    The returned SamplerQNN can be used as a differentiable layer in a hybrid
    learning pipeline.
    """
    # Input qubits and auxiliary qubit for swap test
    total_qubits = input_dim + latent_dim + 1
    circuit = QuantumCircuit(total_qubits, 1)

    # Parameters for input encoding
    input_params = [Parameter(f"x_{i}") for i in range(input_dim)]
    for i, p in enumerate(input_params):
        circuit.ry(p, i)

    # Variational ansatz on latent qubits
    latent_start = input_dim
    ansatz = RealAmplitudes(latent_dim, reps=reps)
    circuit.compose(ansatz, range(latent_start, latent_start + latent_dim), inplace=True)
    weight_params = ansatz.parameters

    # Swap-test to compare encoded input and latent subspace
    aux = input_dim + latent_dim
    circuit.h(aux)
    for i in range(latent_dim):
        circuit.cswap(aux, i, latent_start + i)
    circuit.h(aux)
    circuit.measure(aux, 0)

    sampler = Sampler()

    def interpret(x):
        """
        Interpret the measurement as fidelity.
        `x` is a probability distribution over the single-bit measurement.
        """
        prob_zero = x[0]
        # Fidelity estimate is the probability of measuring 0
        return prob_zero

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        interpret=interpret,
        output_shape=(1,),
        sampler=sampler,
    )
    return qnn
