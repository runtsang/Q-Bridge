"""HybridSamplerQNN: Quantum implementation combining a sampler circuit and a fully connected layer.

This class builds a 3‑qubit variational circuit. The first two qubits implement a
parameterized sampler (analogous to the classical sampler network). The third qubit
acts as an output register whose expectation value of Z corresponds to the
output of a fully connected layer. The circuit is executed on a QASM simulator
and returns a NumPy array containing the expectation value.

Author: <Name>
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter, ParameterVector

class HybridSamplerQNN:
    """
    Quantum neural network combining a sampler and a fully connected layer.
    """

    def __init__(self) -> None:
        # Input parameters for the sampler (two qubits)
        self.input_params = ParameterVector("input", 2)
        # Weight parameters for the sampler (two qubits)
        self.weight_params = ParameterVector("weight", 2)
        # Output weight for the fully connected layer (third qubit)
        self.output_weight = Parameter("output_weight")

        # Build the circuit
        self.circuit = QuantumCircuit(3)
        # Sampler part
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        # Fully connected part
        self.circuit.ry(self.output_weight, 2)
        # Entangle output with sampler
        self.circuit.cx(1, 2)
        # Measure all qubits
        self.circuit.measure_all()

        # Backend and simulation settings
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of the Z observable on qubit 2.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of length 5: [input0, input1, weight0, weight1, output_weight].

        Returns
        -------
        np.ndarray
            Array containing a single expectation value.
        """
        if len(thetas)!= 5:
            raise ValueError("Expected 5 parameters: 2 inputs, 2 sampler weights, 1 output weight.")
        param_dict = {
            self.input_params[0]: thetas[0],
            self.input_params[1]: thetas[1],
            self.weight_params[0]: thetas[2],
            self.weight_params[1]: thetas[3],
            self.output_weight: thetas[4],
        }
        bound_circuit = self.circuit.bind_parameters(param_dict)
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        # Compute expectation of Z on qubit 2
        # Qiskit returns bitstrings with qubit 2 as the first character
        counts_array = np.array(list(counts.values()))
        # Extract bit of qubit 2 from each bitstring
        states = np.array([int(bitstring[0]) for bitstring in counts.keys()])
        probabilities = counts_array / self.shots
        expectation = np.sum((1 - 2 * states) * probabilities)
        return np.array([expectation])

__all__ = ["HybridSamplerQNN"]
