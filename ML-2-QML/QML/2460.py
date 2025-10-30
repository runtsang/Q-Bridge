"""Quantum hybrid fully‑connected layer with a sampler.

The class builds a parameterized quantum circuit that first applies
a fully‑connected style rotation (H + RY) and then a sampler
circuit that produces a probability distribution over two states.
The public API mirrors the classical counterpart: ``run`` returns
the expectation value of the first part, while ``sample`` returns
the sampler probabilities.  The implementation relies on Qiskit
and the Qiskit Machine Learning sampler.

The design keeps the interface compatible with the original FCL
example and enables hybrid experiments where classical and quantum
parameters can be trained jointly.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class HybridFCL:
    """Quantum hybrid fully‑connected layer with a sampler."""

    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        # Backend and simulator
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Fully‑connected part
        self._fc_circuit = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", length=n_qubits)
        self._fc_circuit.h(range(n_qubits))
        self._fc_circuit.barrier()
        for q, t in zip(range(n_qubits), self.theta):
            self._fc_circuit.ry(t, q)
        self._fc_circuit.measure_all()

        # Sampler part
        self._sampler_circuit = QuantumCircuit(2)
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        self._sampler_circuit.ry(inputs2[0], 0)
        self._sampler_circuit.ry(inputs2[1], 1)
        self._sampler_circuit.cx(0, 1)
        self._sampler_circuit.ry(weights2[0], 0)
        self._sampler_circuit.ry(weights2[1], 1)
        self._sampler_circuit.cx(0, 1)
        self._sampler_circuit.ry(weights2[2], 0)
        self._sampler_circuit.ry(weights2[3], 1)

        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self._sampler_circuit,
            input_params=inputs2,
            weight_params=weights2,
            sampler=self.sampler,
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the fully‑connected part of the circuit and return the
        expectation value of the measured qubit(s).

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of rotation angles for the fully‑connected part.

        Returns
        -------
        np.ndarray
            Expectation value as a 1‑element array.
        """
        bind = {self.theta[i]: theta for i, theta in enumerate(thetas)}
        job = qiskit.execute(
            self._fc_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self._fc_circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    def sample(self, inputs: List[List[float]], weights: List[List[float]]) -> np.ndarray:
        """
        Use the sampler QNN to produce a probability distribution.

        Parameters
        ----------
        inputs : List[List[float]]
            List of input parameter vectors of length 2.
        weights : List[List[float]]
            List of weight parameter vectors of length 4.

        Returns
        -------
        np.ndarray
            Array of shape (batch, 2) containing the sampler probabilities.
        """
        probs = self.sampler_qnn.sample(inputs, weight_values=weights)
        return np.array(probs)

    def __repr__(self) -> str:
        return f"HybridFCL(n_qubits={self._fc_circuit.num_qubits}, shots={self.shots})"


__all__ = ["HybridFCL"]
