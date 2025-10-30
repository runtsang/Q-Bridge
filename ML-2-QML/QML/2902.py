"""Quantum implementation of a convolution + sampler network.

This module mirrors the classical `QuantumConvSampler` but uses Qiskit
parameterised circuits. The quantum filter is a quanvolution circuit
followed by a `SamplerQNN` that produces a probability distribution.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Any, Dict


class QuantumConvSampler:
    """
    Quantum counterpart of the classical convolution + sampler network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        backend: Any = None,
        shots: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Convolution (quanvolution) part
        self.n_qubits = kernel_size ** 2
        self.conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.conv_circuit.rx(self.theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(self.n_qubits, 2)
        self.conv_circuit.measure_all()

        # SamplerQNN part
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc2,
            input_params=inputs2,
            weight_params=weights2,
            sampler=sampler,
        )

    def _conv_probability(self, data: Any) -> float:
        """
        Execute the quanvolution circuit on classical data and return the
        average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.conv_circuit)

        total_ones = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            total_ones += ones * val

        return total_ones / (self.shots * self.n_qubits)

    def run(self, data: Any) -> Dict[str, float]:
        """
        Run the full quantum convolution + sampler network.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        dict
            Probability distribution over the sampler's output basis states.
        """
        conv_prob = self._conv_probability(data)

        # Use the convolution probability as two identical inputs to the sampler QNN
        sampler_output = self.sampler_qnn.sample([conv_prob, conv_prob])

        # `sampler_output` is a list of dicts; return the first element
        return sampler_output[0]

    def __call__(self, data: Any) -> Dict[str, float]:
        """Convenience wrapper to allow the object to be called like a function."""
        return self.run(data)


__all__ = ["QuantumConvSampler"]
