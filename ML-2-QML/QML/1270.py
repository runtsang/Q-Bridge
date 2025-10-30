"""Variational quantum circuit for a fully connected layer.

The circuit uses multiple qubits and layers of parameterised Ry rotations
with CNOT entanglement.  The run method binds a flattened list of angles
to the circuit, executes it on a simulator, and returns the expectation
value of Pauli‑Z on all qubits.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from typing import Iterable, List


class FCL:
    """
    Parameterised quantum circuit that emulates a fully connected layer.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the circuit.
    n_layers : int, default 2
        Number of parameterised rotation layers.
    shots : int, default 1024
        Number of shots for simulation.
    backend : qiskit.providers.Backend, optional
        Execution backend.  If None, the Aer qasm_simulator is used.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 2,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameter vector: one Ry per qubit per layer
        self.params = ParameterVector("theta", self.n_qubits * self.n_layers)
        idx = 0
        for _ in range(self.n_layers):
            # Ry rotations
            for q in range(self.n_qubits):
                qc.ry(self.params[idx], q)
                idx += 1
            # Entanglement via CNOT chain
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of rotation angles.  Length must equal
            ``n_qubits * n_layers``.

        Returns
        -------
        np.ndarray
            Expectation value of Pauli‑Z summed over all qubits,
            wrapped in a 1‑D array.
        """
        thetas = list(thetas)
        if len(thetas)!= self.n_qubits * self.n_layers:
            raise ValueError(
                f"Expected {self.n_qubits * self.n_layers} parameters, got {len(thetas)}"
            )
        binding = {self.params[i]: thetas[i] for i in range(len(thetas))}
        bound_qc = self._circuit.bind_parameters(binding)
        job = execute(bound_qc, backend=self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_qc)
        expectation = 0.0
        for state, count in counts.items():
            prob = count / self.shots
            # Qiskit returns bitstrings with qubit 0 as the least significant bit.
            for i, bit in enumerate(state[::-1]):
                z = 1.0 if bit == "0" else -1.0
                expectation += z * prob
        return np.array([expectation])


__all__ = ["FCL"]
