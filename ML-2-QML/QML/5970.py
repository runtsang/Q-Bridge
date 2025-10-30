"""Hybrid quantum convolution–sampler architecture.

The class ``HybridQuantumConvSampler`` fuses the quantum convolution
(called ``QuanvCircuit``) with a Qiskit Machine Learning
``SamplerQNN``.  It mirrors the classical ``HybridConvSampler`` but
provides a fully quantum‑centric pipeline that can be trained with
parameter‑gradient methods supported by Qiskit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class QuanvCircuit:
    """Parameterized quantum convolution circuit."""
    def __init__(self, kernel_size: int, threshold: float,
                 shots: int = 1024, backend: AerSimulator | None = None) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # Add a shallow entangling layer
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the circuit on ``data`` and return the expectation
        value of Z for each qubit as a 1‑D array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        np.ndarray
            Expectation values of shape ``(n_qubits,)``.
        """
        kernel_size = int(np.sqrt(self.n_qubits))
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in data:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self._circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Convert counts to expectation values of Z
        exp = np.zeros(self.n_qubits)
        for bitstring, freq in counts.items():
            weight = freq / self.shots
            for q, bit in enumerate(reversed(bitstring)):
                exp[q] += (1 if bit == '1' else -1) * weight
        return exp

class HybridQuantumConvSampler:
    """Hybrid quantum model that chains a QuanvCircuit with a SamplerQNN."""
    def __init__(self, kernel_size: int = 2, threshold: float = 127,
                 shots: int = 1024, sampler_weights: np.ndarray | None = None) -> None:
        self.conv = QuanvCircuit(kernel_size=kernel_size,
                                 threshold=threshold,
                                 shots=shots)
        # Build input and weight parameters for the sampler
        input_params = ParameterVector("input", self.conv.n_qubits)
        weight_params = ParameterVector("weight", 4)
        qc = QuantumCircuit(self.conv.n_qubits)
        # Simple entangling circuit for sampler
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(circuit=qc,
                                      input_params=input_params,
                                      weight_params=weight_params,
                                      sampler=sampler)

        # If a weight array is supplied, initialize the sampler weights
        if sampler_weights is not None:
            self.sampler_qnn.set_weights(sampler_weights)

    def run(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through the hybrid quantum model.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(2,)`` produced by the
            SamplerQNN.
        """
        conv_out = self.conv.run(data)
        # Convert expectation values to a probability distribution
        probs = self.sampler_qnn.forward(conv_out)
        return probs

    def predict(self, data: np.ndarray) -> int:
        """Return the class index with highest probability."""
        probs = self.run(data)
        return int(np.argmax(probs))

__all__ = ["HybridQuantumConvSampler"]
