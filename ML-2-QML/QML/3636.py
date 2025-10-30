"""Quantum component of HybridQNN.

The class implements a variational circuit that accepts 2 input
parameters and 4 trainable weights.  Measurement statistics of each
qubit are linearly combined to produce a scalar output that mirrors
the fully‑connected read‑out of the classical module.  The circuit
is executed on a qasm simulator but can be swapped for any Qiskit
backend.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler


def HybridQuantumCircuit(backend=None, shots: int = 1024):
    class HybridCircuit:
        """Parameterised quantum circuit that emulates a fully‑connected layer."""
        def __init__(self, backend=None, shots: int = 1024):
            self.n_qubits = 2
            self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots

            # Parameters
            self.inputs = ParameterVector("input", 2)
            self.weights = ParameterVector("weight", 4)

            # Build circuit
            self.circuit = QuantumCircuit(self.n_qubits)
            # Input encoding
            self.circuit.ry(self.inputs[0], 0)
            self.circuit.ry(self.inputs[1], 1)
            self.circuit.cx(0, 1)
            # Variational layers
            for i in range(4):
                self.circuit.ry(self.weights[i], i % 2)
                if i % 2 == 1:
                    self.circuit.cx(0, 1)
            self.circuit.measure_all()

            # Sampler primitive
            self.sampler = Sampler()

        def run(self, thetas: Iterable[float], input_vals: Iterable[float]) -> np.ndarray:
            """
            Execute the circuit with supplied weights and input data.

            Parameters
            ----------
            thetas : Iterable[float]
                4 trainable weight values.
            input_vals : Iterable[float]
                2 input values to encode.

            Returns
            -------
            np.ndarray
                1‑D array containing the scalar output.
            """
            # Bind parameters
            param_binds = [
                {self.inputs[0]: input_vals[0], self.inputs[1]: input_vals[1]},
                {self.weights[i]: thetas[i] for i in range(4)}
            ]

            # Run simulation
            job = self.sampler.run(self.circuit, parameter_binds=param_binds, shots=self.shots)
            result = job.result()
            counts = result.get_counts(self.circuit)

            # Compute expectation values of Z on each qubit
            expectations = []
            for q in range(self.n_qubits):
                exp = 0.0
                for state, cnt in counts.items():
                    # state string is little‑endian; reverse for readability
                    bit = int(state[::-1][q])
                    exp += cnt * (1 if bit == 0 else -1)
                exp /= self.shots
                expectations.append(exp)

            # Simple linear read‑out: tanh(z0 - z1)
            out = np.tanh(expectations[0] - expectations[1])
            return np.array([out])

    return HybridCircuit()
