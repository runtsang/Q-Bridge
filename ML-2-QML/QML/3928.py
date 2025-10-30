"""Hybrid quantum sampler network with an embedded fully‑connected layer.

The quantum implementation constructs:
- A 2‑qubit sampler circuit with ParameterVectors for input and weight
  parameters.
- A 1‑qubit fully‑connected layer that measures the expectation of
  the Pauli‑Z operator after a Ry rotation controlled by the sampler
  output.

The run method accepts a 2‑dimensional input vector and a 4‑dimensional
weight vector, executes the composite circuit on the Aer qasm simulator,
and returns a 2‑element NumPy array containing the expectation values
of the two layers.  These values can be fed into a classical classifier
or used for further quantum processing.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import Backend


class SamplerQNN:
    """
    Quantum sampler network with an integrated fully‑connected layer.
    """

    def __init__(self,
                 backend: Backend | None = None,
                 shots: int = 1024) -> None:
        # Default backend
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # ---- Sampler circuit (2 qubits) ----
        self.qc = QuantumCircuit(2)
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        self.qc.ry(self.inputs[0], 0)
        self.qc.ry(self.inputs[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[0], 0)
        self.qc.ry(self.weights[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(self.weights[2], 0)
        self.qc.ry(self.weights[3], 1)
        self.qc.measure_all()

        # ---- Fully‑connected layer (1 qubit) ----
        self.fc_qc = QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.fc_qc.h(0)
        self.fc_qc.ry(self.theta, 0)
        self.fc_qc.measure_all()

    def _expectation_z(self, counts: dict[str, int]) -> float:
        """Compute expectation value of Z on the first qubit."""
        total = sum(counts.values())
        exp = 0.0
        for state, cnt in counts.items():
            prob = cnt / total
            # state string is little‑endian with qubit 1 as most significant bit
            qubit0 = int(state[0])  # qubit 1
            exp += (1 if qubit0 == 0 else -1) * prob
        return exp

    def _expectation_z_single(self, counts: dict[str, int]) -> float:
        """Compute expectation of Z on the single‑qubit FC circuit."""
        total = sum(counts.values())
        exp = 0.0
        for state, cnt in counts.items():
            prob = cnt / total
            qubit0 = int(state[0])  # single qubit
            exp += (1 if qubit0 == 0 else -1) * prob
        return exp

    def run(self,
            inputs: np.ndarray | list[float],
            weights: np.ndarray | list[float]) -> np.ndarray:
        """
        Execute the composite circuit and return expectation values.

        Parameters
        ----------
        inputs : array‑like, shape (2,)
            Input parameters for the sampler circuit.
        weights : array‑like, shape (4,)
            Weight parameters for the sampler circuit.

        Returns
        -------
        np.ndarray
            Array of shape (2,) containing the expectation of the sampler
            (first qubit) and the expectation of the fully‑connected layer.
        """
        # Bind parameters
        bind_dict = {
            self.inputs[0]: float(inputs[0]),
            self.inputs[1]: float(inputs[1]),
            self.weights[0]: float(weights[0]),
            self.weights[1]: float(weights[1]),
            self.weights[2]: float(weights[2]),
            self.weights[3]: float(weights[3]),
        }

        # Execute sampler circuit
        job = execute(self.qc,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[bind_dict])
        result = job.result()
        counts = result.get_counts(self.qc)

        # Expectation from sampler (Z on first qubit)
        exp_sampler = self._expectation_z(counts)

        # Use sampler expectation as theta for FC circuit
        fc_bind = {self.theta: exp_sampler}
        fc_job = execute(self.fc_qc,
                         self.backend,
                         shots=1,
                         parameter_binds=[fc_bind])
        fc_counts = fc_job.result().get_counts(self.fc_qc)
        exp_fc = self._expectation_z_single(fc_counts)

        return np.array([exp_sampler, exp_fc])


__all__ = ["SamplerQNN"]
