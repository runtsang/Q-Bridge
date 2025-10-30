"""Hybrid quantum convolutional sampler.

This module implements a quantum‑enhanced convolutional filter coupled with
a SamplerQNN for probabilistic classification.  The quantum circuit
applies a parameterized rotation to each qubit representing a pixel,
followed by a random layer and measurement.  The resulting feature
(average probability of measuring |1>) is fed into a SamplerQNN
parameterized by additional weights, and a StatevectorSampler is used
to obtain class probabilities.

The class ``HybridConvSampler`` can be instantiated directly or via the
``Conv`` helper to maintain API compatibility.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler


class HybridConvSampler:
    """
    Quantum hybrid convolutional sampler.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (must be square).
    threshold : float
        Threshold used to binarize pixel values before parameter binding.
    backend : qiskit.providers.Provider, optional
        Qiskit backend for execution.  Defaults to Aer qasm_simulator.
    shots : int, optional
        Number of shots for the convolution circuit.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 backend=None,
                 shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Convolution circuit
        self._conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._conv_circuit.rx(self.theta[i], i)
        self._conv_circuit.barrier()
        self._conv_circuit += random_circuit(self.n_qubits, 2)
        self._conv_circuit.measure_all()

        # Sampler circuit (SamplerQNN structure)
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        self._sampler_circuit = qc
        self._inputs = inputs
        self._weights = weights

        # Sampler primitive
        self._sampler = StatevectorSampler()

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the hybrid quantum sampler on 2‑D data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Class probabilities of shape (2,).
        """
        # Prepare bindings for the convolution circuit
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0)
                    for i, val in enumerate(row)}
            param_binds.append(bind)

        # Execute convolution circuit
        job = execute(
            self._conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._conv_circuit)

        # Compute average probability of |1> over all qubits
        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * count
        avg_prob = total_ones / (self.shots * self.n_qubits)

        # Bind sampler inputs (first input is avg_prob, second is dummy)
        sampler_bind = {
            self._inputs[0]: avg_prob,
            self._inputs[1]: 0.0,
            self._weights[0]: 0.0,
            self._weights[1]: 0.0,
            self._weights[2]: 0.0,
            self._weights[3]: 0.0,
        }

        # Execute sampler circuit
        sampler_job = self._sampler.run(
            self._sampler_circuit,
            parameter_binds=[sampler_bind],
        )
        sampler_counts = sampler_job.get_counts(self._sampler_circuit)

        # Convert counts to probabilities
        total = sum(sampler_counts.values())
        probs = {state: cnt / total for state, cnt in sampler_counts.items()}

        # Map 2‑qubit outputs to 2 classes:
        # class 0: bitstring starts with '0', class 1: starts with '1'
        p0 = sum(p for state, p in probs.items() if state[0] == '0')
        p1 = sum(p for state, p in probs.items() if state[0] == '1')
        return np.array([p0, p1])

def Conv() -> HybridConvSampler:
    """
    Return a quantum hybrid conv‑sampler instance.
    """
    return HybridConvSampler()

__all__ = ["Conv", "HybridConvSampler"]
