"""
Hybrid sampler – quantum side.

`SamplerQNNGen224` is a thin wrapper around
`qiskit_machine_learning.neural_networks.SamplerQNN` that
creates a depth‑controlled variational ansatz.  The circuit
encodes inputs via Ry rotations, interleaves variational Ry
rotations with CZ entanglers, and measures in the Pauli‑Z basis.
The module exposes convenient helpers for probability
evaluation and sampling, with a default sampler that can
be swapped for a local simulator or a real device.

Key features
------------
* Depth‑controlled ansatz – `depth` controls the number of
  variational layers.
* Pauli‑Z observables per qubit – convenient for probability
  estimation.
* `sample` method that draws up to 224 samples in a single call.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as QiskitSampler


class SamplerQNNGen224(SamplerQNN):
    """
    Quantum variational sampler powered by a depth‑controlled ansatz.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input vector (will be encoded
        with Ry rotations).
    depth : int
        Number of variational layers; each layer adds a Ry per qubit
        and a CZ entangler between neighbours.
    num_qubits : int | None
        Number of qubits in the circuit; defaults to `input_dim`.
    sampler : qiskit.primitives.Sampler | None
        Primitive used to execute the circuit.  If ``None``, a
        default local simulator is instantiated.
    """

    def __init__(
        self,
        input_dim: int = 2,
        depth: int = 2,
        num_qubits: int | None = None,
        sampler: QiskitSampler | None = None,
    ) -> None:
        if num_qubits is None:
            num_qubits = input_dim

        # Parameter vectors for inputs and variational angles
        input_params = ParameterVector("input", input_dim)
        weight_params = ParameterVector("theta", num_qubits * depth)

        # Construct the ansatz
        circuit = QuantumCircuit(num_qubits)

        # Data encoding
        for idx, qubit in enumerate(range(num_qubits)):
            circuit.ry(input_params[idx], qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weight_params[idx], qubit)
                idx += 1
            # Entanglement: CZ between adjacent qubits
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Pauli‑Z observables – one per qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        if sampler is None:
            sampler = QiskitSampler()

        super().__init__(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            sampler=sampler,
            observables=observables,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_probability_distribution(
        self,
        inputs: np.ndarray | list[float] | tuple[float,...],
        weights: np.ndarray | list[float] | tuple[float,...],
    ) -> np.ndarray:
        """
        Evaluate the expectation values of the Pauli‑Z observables,
        which can be interpreted as a probability distribution over
        the measurement outcomes.

        Parameters
        ----------
        inputs : array-like, shape (batch, input_dim)
            Classical inputs to the circuit.
        weights : array-like, shape (batch, weight_dim)
            Variational angles produced by the classical network.

        Returns
        -------
        np.ndarray
            Shape (batch, num_qubits) – expectation values in [-1, 1].
        """
        return self.forward(inputs, weights)

    def sample(
        self,
        inputs: np.ndarray | list[float] | tuple[float,...],
        weights: np.ndarray | list[float] | tuple[float,...],
        num_samples: int = 224,
    ) -> np.ndarray:
        """
        Draw samples from the quantum circuit’s output distribution.

        Parameters
        ----------
        inputs : array-like, shape (batch, input_dim)
            Classical inputs.
        weights : array-like, shape (batch, weight_dim)
            Variational angles.
        num_samples : int
            Number of samples to draw per batch element.

        Returns
        -------
        np.ndarray
            Shape (batch, num_samples) – sampled bitstrings
            represented as integers.
        """
        # The underlying SamplerQNN expects a batch of parameters.
        # We flatten them into a single vector per batch element.
        batch_params = np.concatenate([np.asarray(inputs), np.asarray(weights)], axis=1)
        # Execute the sampler
        result = self.sampler.run(
            self.circuit,
            parameter_binds=[dict(
                zip(self.input_params + self.weight_params,
                    batch_params[i]))
            for i in range(batch_params.shape[0])]
        ).result()

        # Retrieve the measurement counts
        counts = result.get_counts()

        # Convert counts to probability distribution
        probs = {k: v / result.get_counts().total_counts for k, v in counts.items()}

        # Sample from the distribution
        outcomes = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        return np.random.choice(outcomes, size=(batch_params.shape[0], num_samples), p=probabilities)

    def set_weights_from_tensor(self, weights: np.ndarray | torch.Tensor) -> None:
        """
        Utility to set the weight parameters directly from a
        NumPy array or PyTorch tensor.  This is handy when the
        classical network outputs a tensor that should be fed
        into the quantum sampler without manual conversion.

        Parameters
        ----------
        weights : array-like or torch.Tensor
            Shape (batch, weight_dim)
        """
        # Convert to NumPy if necessary
        if hasattr(weights, "numpy"):
            weights = weights.numpy()
        self.weight_params = ParameterVector("theta", weights.size)
        # In practice, the sampler will bind these parameters during
        # execution; this method only updates the internal reference.

__all__ = ["SamplerQNNGen224"]
