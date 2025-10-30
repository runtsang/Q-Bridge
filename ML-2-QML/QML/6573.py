from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler


class HybridLayer:
    """
    HybridLayer implements a quantum variational sampler circuit that mimics
    the classical architecture of HybridLayer. It accepts an input parameter
    vector and a weight vector, builds a parameterized circuit, and uses
    a StatevectorSampler to compute the expectation value.
    """

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Define parameters
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", 4)

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Input encoding
        for i in range(n_qubits):
            self.circuit.ry(self.input_params[i], i)
        # Entangling layer
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Variational layer
        for i in range(n_qubits):
            self.circuit.ry(self.weight_params[i], i)
        # Measurement
        self.circuit.measure_all()

        self.sampler = StatevectorSampler()

    def run(self, thetas):
        """
        Execute the circuit with the provided parameters.
        `thetas` must be a tuple (input_thetas, weight_thetas).
        Returns the expectation value as a NumPy array.
        """
        input_thetas, weight_thetas = thetas
        param_binds = [
            {self.input_params[i]: input_thetas[i] for i in range(self.n_qubits)},
            {self.weight_params[i]: weight_thetas[i] for i in range(4)},
        ]
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        counts_arr = np.array(list(counts.values()))
        # Convert bitstrings to integer states (reverse bit order to match Qiskit convention)
        states_arr = np.array([int(state[::-1], 2) for state in counts.keys()])
        probs = counts_arr / self.shots
        expectation = np.sum(states_arr * probs)
        return np.array([expectation])


__all__ = ["HybridLayer"]
