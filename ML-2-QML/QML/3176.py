"""Quantum sampler for SamplerQNNGen108.

This module defines SamplerQNNGen108Quantum, a parameterised quantum
circuit that accepts both input and weight parameters.  The circuit
consists of a two‑qubit sampler (as in SamplerQNN.py) followed by a
single‑qubit fully‑connected layer (as in FCL.py).  The `run` method
binds parameters, executes the circuit on a chosen backend, and
returns the expectation values of Z on the first and second qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Iterable, Optional

class SamplerQNNGen108Quantum:
    """Parameterised quantum circuit for the hybrid sampler.

    Parameters
    ----------
    backend : Optional[Backend], default Aer.get_backend('qasm_simulator')
        Backend to execute the circuit.
    shots : int, default 1024
        Number of shots for sampling.
    """

    def __init__(self, backend: Optional[object] = None, shots: int = 1024) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Define parameters
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.fc_param = ParameterVector("fc", 1)

        # Build the circuit
        self._build_circuit()

    def _build_circuit(self) -> None:
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Sampler part – two qubits (0 and 1)
        self.circuit.ry(self.input_params[0], qr[0])
        self.circuit.ry(self.input_params[1], qr[1])
        self.circuit.cx(qr[0], qr[1])

        # Weight layers
        self.circuit.ry(self.weight_params[0], qr[0])
        self.circuit.ry(self.weight_params[1], qr[1])
        self.circuit.cx(qr[0], qr[1])

        self.circuit.ry(self.weight_params[2], qr[0])
        self.circuit.ry(self.weight_params[3], qr[1])

        # Fully‑connected layer – single qubit (2)
        self.circuit.h(qr[2])
        self.circuit.ry(self.fc_param[0], qr[2])
        self.circuit.barrier()

        # Measure all
        self.circuit.measure(qr, cr)

    def run(self, params: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        params : Iterable[float]
            Sequence of length 7: [input0, input1, w0, w1, w2, w3, fc].

        Returns
        -------
        expectation : np.ndarray
            Shape (2,) – expectation values of Z on qubits 0 and 1.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.shape[1]!= 7:
            raise ValueError("Expected 7 parameters: 2 inputs, 4 weights, 1 fc.")

        expectations = []
        for row in params:
            bind_dict = {
                self.input_params[0]: row[0],
                self.input_params[1]: row[1],
                self.weight_params[0]: row[2],
                self.weight_params[1]: row[3],
                self.weight_params[2]: row[4],
                self.weight_params[3]: row[5],
                self.fc_param[0]: row[6],
            }
            bound_circ = self.circuit.bind_parameters(bind_dict)

            job = execute(bound_circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circ)

            exp0, exp1 = 0.0, 0.0
            total = 0
            for bitstring, count in counts.items():
                bits = list(reversed(bitstring))
                z0 = 1 if bits[0] == "0" else -1
                z1 = 1 if bits[1] == "0" else -1
                exp0 += z0 * count
                exp1 += z1 * count
                total += count

            exp0 /= total
            exp1 /= total
            expectations.append([exp0, exp1])

        return np.array(expectations)

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying QuantumCircuit object."""
        return self.circuit

__all__ = ["SamplerQNNGen108Quantum"]
