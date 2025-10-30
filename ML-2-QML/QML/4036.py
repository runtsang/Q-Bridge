"""Hybrid Quantum Sampler QNN that mirrors the classical implementation and provides a full quantum backend."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector

class HybridSamplerQNN:
    """
    Quantum implementation of the hybrid sampler network.
    The circuit is parameterised by two input angles and four weight angles.
    The run method returns the expectation value of the Pauli‑Z operator on qubit 0
    for each input sample.
    """

    def __init__(self, backend: str = "aer_simulator", shots: int = 1000):
        self.backend = Aer.get_backend(backend)
        self.shots = shots

        # Define parameter vectors
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Build circuit template
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Fixed weight values for simulation (can be learned externally)
        self.weights = np.random.randn(4)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of input thetas.

        Args:
            thetas: ndarray of shape (batch, 2) containing the input angles.

        Returns:
            ndarray of shape (batch, 1) containing the expectation values.
        """
        expectations = []
        for theta0, theta1 in thetas:
            # Bind input and weight parameters
            bind_dict = {
                self.input_params[0]: theta0,
                self.input_params[1]: theta1,
            }
            for i, w in enumerate(self.weights):
                bind_dict[self.weight_params[i]] = w

            bound_circ = self.circuit.bind_parameters(bind_dict)

            # Execute on the simulator
            job = execute(
                bound_circ,
                backend=self.backend,
                shots=self.shots,
            )
            result = job.result()
            counts = result.get_counts(bound_circ)
            probs = {state: cnt / self.shots for state, cnt in counts.items()}

            # Compute expectation of Z on qubit 0
            exp = 0.0
            for state, p in probs.items():
                if state[0] == "0":
                    exp += p
                else:
                    exp -= p
            expectations.append([exp])
        return np.array(expectations)

__all__ = ["HybridSamplerQNN"]
