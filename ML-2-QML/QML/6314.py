"""
Quantum implementation of a fully‑connected layer using a parameterized
circuit.  The circuit applies a Hadamard gate, a parameterized Ry rotation,
an entangling CX ladder, and finally measures all qubits.  The expectation
value of the Pauli‑Z operator on the first qubit is returned as a
classical feature.

The module is intentionally lightweight so that it can be swapped
in for the classical HybridFCL in hybrid experiments.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer


class QuantumFCL:
    """
    Quantum fully‑connected layer implemented with Qiskit.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits to use.  One qubit is sufficient for a single
        parameterized rotation, but more qubits allow the addition of
        entanglement and random layers for richer feature maps.
    shots : int, default=1024
        Number of shots for the circuit execution.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit on.  If None, the Aer qasm simulator
        is used.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024,
                 backend: qiskit.providers.Backend | None = None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a parameterized circuit with entanglement."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Create a superposition
        qc.h(qr)

        # Parameterized rotation on each qubit
        qc.ry(self.theta, qr)

        # Entangling CX ladder
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])

        # Measure all qubits
        qc.measure(qr, cr)
        return qc

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Execute the circuit for each theta and return the expectation
        value of the Pauli‑Z operator on the first qubit.

        Parameters
        ----------
        thetas : list[float]
            List of rotation angles.

        Returns
        -------
        np.ndarray
            Array of shape (len(thetas), 1) containing the expectation
            values.
        """
        param_binds = [{self.theta: theta} for theta in thetas]
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        expectations = []

        for theta, bind in zip(thetas, param_binds):
            counts = result.get_counts(self.circuit, parameter_binds=[bind])
            counts = {int(k[::-1], 2): v for k, v in counts.items()}
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()))
            # Expectation of Z on first qubit: (+1 for |0⟩, -1 for |1⟩)
            z_exp = np.sum((2 * (states >> (self.n_qubits - 1)) - 1) * probs)
            expectations.append(z_exp)

        return np.array(expectations).reshape(-1, 1)


__all__ = ["QuantumFCL"]
