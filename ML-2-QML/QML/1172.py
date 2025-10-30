"""Quantum implementation of a fully connected layer.

This module builds a variational circuit that maps a vector of classical
parameters (thetas) to the expectation value of Pauli‑Z on each qubit.
The circuit contains an H‑gate on every qubit, a parameterised RY gate,
and a chain of CX gates that entangle neighbouring qubits.  The
expectation values can be used as a quantum analogue of a neural network
layer.  The class exposes a `run` method that accepts a flat list of
parameters and returns a NumPy array of expectation values.

The circuit is compatible with the original API: `FCL()` returns an
instance with a `run` method.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from typing import Iterable


class QuantumFullyConnectedLayer:
    """Variational circuit that emulates a fully‑connected layer."""

    def __init__(
        self,
        n_qubits: int = 1,
        backend: Backend | None = None,
        shots: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        n_qubits:
            Number of qubits in the circuit.
        backend:
            Qiskit backend.  If ``None`` the Aer qasm simulator is used.
        shots:
            Number of shots for the measurement.
        """
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vector – one RY angle per qubit
        self.theta = ParameterVector("theta", self.n_qubits)

        # Build the circuit
        self._circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self._circuit.h(range(self.n_qubits))
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta[i], i)
        # Entanglement chain
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)
        self._circuit.barrier()
        self._circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of Pauli‑Z on each qubit.

        Parameters
        ----------
        thetas:
            List of rotation angles, one per qubit.  The list must have
            length ``n_qubits``.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError("theta vector length must match n_qubits")

        # Bind parameters
        bound_circ = self._circuit.bind_parameters(
            {self.theta[i]: float(thetas[i]) for i in range(self.n_qubits)}
        )

        job = execute(
            bound_circ,
            self.backend,
            shots=self.shots,
            memory=True,
        )
        result = job.result()
        counts = result.get_counts(bound_circ)

        # Convert counts to expectation values of Pauli‑Z
        exp_vals = np.zeros(self.n_qubits, dtype=float)
        for state, cnt in counts.items():
            prob = cnt / self.shots
            # Binary string: most significant bit is qubit 0
            for i, bit in enumerate(reversed(state)):
                z = 1 if bit == "0" else -1  # |0> -> +1, |1> -> -1
                exp_vals[i] += z * prob

        return exp_vals

    def parameter_shift_gradient(
        self, thetas: Iterable[float], shift: float = np.pi / 2
    ) -> np.ndarray:
        """
        Estimate the gradient of the expectation vector w.r.t. the parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas:
            Current parameter vector.
        shift:
            Shift value for the parameter‑shift rule.  Default is π/2.
        """
        grads = np.zeros((self.n_qubits, self.n_qubits))
        for k in range(self.n_qubits):
            # Positive shift
            thetas_plus = list(thetas)
            thetas_plus[k] += shift
            exp_plus = self.run(thetas_plus)

            # Negative shift
            thetas_minus = list(thetas)
            thetas_minus[k] -= shift
            exp_minus = self.run(thetas_minus)

            grads[:, k] = 0.5 * (exp_plus - exp_minus)

        return grads

def FCL() -> QuantumFullyConnectedLayer:
    """Return a quantum fully‑connected layer with a single qubit."""
    return QuantumFullyConnectedLayer(n_qubits=1, shots=1000)

__all__ = ["FCL"]
