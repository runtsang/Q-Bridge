"""Quantum autoencoder with a variational circuit that learns a latent representation.

The circuit comprises a feature map (RawFeatureVector) to encode the input data,
a RealAmplitudes ansatz to model the latent code, and a measurement that
returns a probability distribution over all basis states.  The interpret
function transforms these probabilities into a reconstruction vector of the
same dimensionality as the input.  This simple example demonstrates how to
embed a classical autoencoder into a hybrid quantum circuit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RawFeatureVector, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler

def Autoencoder__gen240(
    input_dim: int,
    *,
    latent_dim: int = 3,
    ansatz_reps: int = 5,
    sampler: Sampler | None = None,
) -> SamplerQNN:
    """Return a hybrid quantum autoencoder."""
    algorithm_globals.random_seed = 42
    if sampler is None:
        sampler = Sampler()
    # Feature map to embed classical data
    feature_map = RawFeatureVector(input_dim)
    # Ansatz for latent space
    ansatz = RealAmplitudes(latent_dim, reps=ansatz_reps)
    # Construct the full circuit: feature map -> ansatz
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    # Measure all qubits to obtain the probability distribution
    circuit.measure_all()
    # Interpret function: convert probability distribution into a reconstruction vector
    def interpret(x: np.ndarray) -> np.ndarray:
        # x is the probability vector of length 2**latent_dim
        probs = x
        recon = np.zeros(input_dim)
        for idx, p in enumerate(probs):
            # Map the binary index to a weight between 0 and 1
            bits = format(idx, f"0{latent_dim}b")
            weight = sum(int(b) for b in bits) / latent_dim
            recon += weight * p
        norm = np.linalg.norm(recon)
        if norm > 0:
            recon = recon / norm
        return recon
    # Build the QNN
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],  # No classical parameters in this simple example
        weight_params=ansatz.parameters,
        interpret=interpret,
        output_shape=(input_dim,),
        sampler=sampler,
    )
    return qnn
