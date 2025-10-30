"""
QuantumAutoencoderGen – Quantum swap‑test autoencoder with RealAmplitudes ansatz.

The circuit encodes classical data into a feature map, variationally
prepares a latent+trash sub‑register, and measures a swap‑test
to extract a latent representation.  The class exposes a
`forward` method that accepts a batch of floats and returns a
reconstructed batch, mirroring the classical API for easy hybrid
experiments.
"""

import numpy as np
import torch
from typing import Iterable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.utils import algorithm_globals

# Ensure reproducible quantum behaviour
algorithm_globals.random_seed = 42


class QuantumAutoencoderGen:
    """
    Quantum autoencoder that implements a swap‑test based latent extraction.
    Parameters
    ----------
    num_latent : int
        Number of latent qubits that will be extracted by the swap test.
    num_trash : int
        Number of auxiliary qubits used to increase expressivity.
    num_features : int
        Dimensionality of the classical input vector.
    reps : int, optional
        Number of repetitions for the RealAmplitudes ansatz.
    """

    def __init__(self, num_latent: int, num_trash: int, num_features: int, reps: int = 3):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.num_features = num_features
        self.reps = reps

        # Build the underlying circuit and SamplerQNN
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=self._interpret,
            output_shape=self.num_features,
            sampler=self.sampler,
        )

    def _build_circuit(self):
        """
        Constructs a circuit that:
        1. Encodes the input data with a RealAmplitudes feature map.
        2. Applies a variational ansatz on the latent+trash qubits.
        3. Performs a swap test with an auxiliary qubit.
        """
        # Qubit registers
        data_q = QuantumRegister(self.num_features, "data")
        lt_q = QuantumRegister(self.num_latent + self.num_trash, "lt")
        aux_q = QuantumRegister(1, "aux")
        circuit = QuantumCircuit(data_q, lt_q, aux_q)

        # Feature map on data qubits
        feature_map = RealAmplitudes(self.num_features, reps=self.reps)
        circuit.compose(feature_map, data_q, inplace=True)

        # Variational ansatz on latent+trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        circuit.compose(ansatz, lt_q, inplace=True)

        # Swap test with auxiliary qubit
        circuit.h(aux_q[0])
        for i in range(self.num_trash):
            circuit.cswap(aux_q[0], lt_q[self.num_latent + i], lt_q[self.num_latent + self.num_trash + i])
        circuit.h(aux_q[0])
        circuit.measure(aux_q[0], ClassicalRegister(1, "c"))

        # Input and weight parameters
        input_params = feature_map.parameters
        weight_params = ansatz.parameters
        return circuit, input_params, weight_params

    def _interpret(self, x: np.ndarray) -> np.ndarray:
        """
        Interpret the sampler output as a vector of expectation values
        for each feature qubit.  The sampler returns a probability
        distribution over 0/1 outcomes; the expectation of Z is
        1 - 2 * p(1).
        """
        probs = np.array(x)
        # Convert to expectation values of Z
        return 1.0 - 2.0 * probs

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Feed a batch of classical data through the quantum autoencoder.
        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, num_features) with values in the range [-1, 1].
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch, num_features).
        """
        # Rescale to [0, π] for the RealAmplitudes feature map
        batch = data.cpu().numpy()
        batch = (batch + 1.0) * np.pi

        # Run the SamplerQNN
        outputs = self.qnn(batch)
        return torch.tensor(outputs, dtype=torch.float32, device=data.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Extract the latent representation by performing the swap test
        and measuring the auxiliary qubit.  The output is a binary
        vector of shape (batch, 1) indicating the swap‑test result.
        """
        batch = data.cpu().numpy()
        batch = (batch + 1.0) * np.pi
        # Run the circuit with the sampler to get measurement outcomes
        result = self.sampler.run(self.circuit, shots=1024, parameter_binds=[dict(zip(self.input_params, batch[i])) for i in range(len(batch))])
        probs = result.quasi_dists
        # Convert to expectation values of the auxiliary qubit
        aux_expectation = []
        for dist in probs:
            p0 = dist.get((0,)) or 0.0
            p1 = dist.get((1,)) or 0.0
            aux_expectation.append(1.0 - 2.0 * p1)
        return torch.tensor(aux_expectation, dtype=torch.float32, device=data.device)

__all__ = ["QuantumAutoencoderGen"]
