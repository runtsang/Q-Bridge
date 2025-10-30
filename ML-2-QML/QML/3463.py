"""Hybrid fully‑connected layer – quantum implementation.

The quantum side constructs a 2‑qubit parameterized circuit that
mirrors the sampler architecture from the original reference.
It uses a `StatevectorSampler` to compute both a probability
distribution over the basis states and an expectation value
for the first qubit (Z‑measurement).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

__all__ = ["HybridFCL"]


def HybridFCL():
    class HybridFCLQuantum:
        """Quantum circuit with a statevector sampler."""

        def __init__(self, backend=None, shots: int = 1024) -> None:
            self.backend = backend or Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.n_qubits = 2

            # Parameter vectors
            self.input_params = ParameterVector("input", 2)
            self.weight_params = ParameterVector("weight", 4)

            # Build circuit
            self.circuit = QuantumCircuit(self.n_qubits)
            self.circuit.ry(self.input_params[0], 0)
            self.circuit.ry(self.input_params[1], 1)
            self.circuit.cx(0, 1)
            self.circuit.ry(self.weight_params[0], 0)
            self.circuit.ry(self.weight_params[1], 1)
            self.circuit.cx(0, 1)
            self.circuit.ry(self.weight_params[2], 0)
            self.circuit.ry(self.weight_params[3], 1)
            self.circuit.measure_all()

            # Sampler for probability distribution
            self.sampler = Sampler()

        def run(self, thetas: list[float]) -> tuple[np.ndarray, np.ndarray]:
            """
            Parameters:
                thetas (list[float]): length 6
                - first 2: input parameters
                - next 4: weight parameters

            Returns:
                probs (np.ndarray): probability of each 2‑qubit basis state
                expectation (np.ndarray): expectation of Z on the first qubit
            """
            # Bind parameters
            param_binds = {
                self.input_params[0]: thetas[0],
                self.input_params[1]: thetas[1],
                self.weight_params[0]: thetas[2],
                self.weight_params[1]: thetas[3],
                self.weight_params[2]: thetas[4],
                self.weight_params[3]: thetas[5],
            }
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[param_binds],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)

            # Compute probabilities
            probs_dict = {
                state: count / self.shots for state, count in counts.items()
            }
            probs = np.array(
                [probs_dict.get(f"{i:02b}", 0.0) for i in range(4)]
            )

            # Expectation of Z on qubit 0
            expectation = 0.0
            for state, p in probs_dict.items():
                # qubit 0 is the leftmost bit in the state string
                first_bit = int(state[0])
                z = 1 if first_bit == 0 else -1
                expectation += z * p
            return probs, np.array([expectation])

    return HybridFCLQuantum()
