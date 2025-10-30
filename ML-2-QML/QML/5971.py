"""Quantum circuit implementation for the FCL module.

The circuit implements a simple variational ansatz that
maps each input parameter to an Ry rotation on a separate qubit.
After the rotations a controlled‑X entangling layer is added,
and the expectation value of the Pauli‑Z operator on the first
qubit is returned.  The circuit can be executed on any Qiskit
backend and is fully compatible with the `FCL` class defined
in the classical module above.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class FCL:
    """Variational circuit used as a quantum augmentation of a
    classical fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / parameters in the ansatz.
    backend : qiskit.providers.BaseBackend, optional
        Backend to execute the circuit on.  If ``None`` a
        local ``qasm_simulator`` is used.
    shots : int, optional
        Number of shots for the execution.  Default is 1024.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Create a parameterized circuit.
        self._circuit = QuantumCircuit(n_qubits)
        self.params = [Parameter(f"theta_{i}") for i in range(n_qubits)]

        # Ansatz: Ry rotations on each qubit.
        for i, p in enumerate(self.params):
            self._circuit.ry(p, i)

        # Simple entanglement: a chain of CNOTs.
        for i in range(n_qubits - 1):
            self._circuit.cx(i, i + 1)

        # Measure all qubits.
        self._circuit.measure_all()

    def run(self, thetas: list[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : list[float]
            Parameters to bind to the ``Ry`` gates.  Must have length
            ``n_qubits``.
        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value of the
            computational basis measurement (treated as an integer
            value).  The array shape is ``(1,)`` for consistency
            with the classical layer.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {len(thetas)}"
            )
        bound = {p: t for p, t in zip(self.params, thetas)}
        bound_circuit = self._circuit.bind_parameters(bound)

        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert counts to probabilities and integer states.
        probs = np.array([counts.get(b, 0) for b in sorted(counts)]) / self.shots
        states = np.array([int(b, 2) for b in sorted(counts)])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def circuit(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit."""
        return self._circuit

__all__ = ["FCL"]
