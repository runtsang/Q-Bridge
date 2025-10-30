"""Quantum sampler network with configurable qubits, depth, and sampling backend."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.fake_provider import FakeVigo
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.utils import QuantumInstance


class SamplerQNN:
    """
    Quantum sampler with adjustable depth and backend.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    depth : int, default 2
        Number of entangling layers.
    backend : str, optional
        Backend name; if None, a default FakeVigo is used.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 2, backend: str | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend_name = backend or "FakeVigo"

        # Parameter vectors
        self.input_params = ParameterVector("input", num_qubits)
        self.weight_params = ParameterVector("weight", num_qubits * depth * 2)

        # Build circuit
        self.circuit = self._build_circuit()

        # Sampler primitive
        self.sampler = QiskitSampler(QuantumInstance(backend=FakeVigo()))

        # Wrap with Qiskit ML SamplerQNN
        self.qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input rotations
        for i in range(self.num_qubits):
            qc.ry(self.input_params[i], i)
        # Entangling layers
        for d in range(self.depth):
            # Parameterised single‑qubit rotations
            for i in range(self.num_qubits):
                idx = d * self.num_qubits * 2 + i * 2
                qc.ry(self.weight_params[idx], i)
                qc.rz(self.weight_params[idx + 1], i)
            # Entangling CNOTs
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def sample(self, inputs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """
        Return measurement probabilities for the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (n_samples, num_qubits) with values in [-π, π].
        weights : np.ndarray, optional
            Shape (num_qubits * depth * 2,) of rotation angles. If None, random weights are used.

        Returns
        -------
        np.ndarray
            Probability distribution over 2^num_qubits basis states.
        """
        if weights is None:
            weights = np.random.uniform(-np.pi, np.pi, size=self.weight_params.size)
        bound_circuit = self.circuit.bind_parameters(
            {**{p: w for p, w in zip(self.weight_params, weights)},
             **{p: x for p, x in zip(self.input_params, inputs.T)}}
        )
        result = self.sampler.run(bound_circuit).result()
        probs = result.get_counts()
        # Convert dict to vector
        num_states = 2 ** self.num_qubits
        prob_vec = np.zeros(num_states)
        for bitstring, count in probs.items():
            idx = int(bitstring, 2)
            prob_vec[idx] = count
        prob_vec /= prob_vec.sum()
        return prob_vec

__all__ = ["SamplerQNN"]
