"""Variational quantum fully‑connected layer with multi‑qubit support.

Features
--------
* Parameterised Ry rotations per qubit
* Optional entangling CNOT chain
* Expectation values of Pauli‑Z measured on each qubit
* Parameter‑shift gradient computation
* Back‑end agnostic (Qiskit Aer, local simulator, or any Qiskit provider)
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers import Backend
from qiskit.result import Result
from typing import Iterable, Optional, List

class FCL:
    """
    Variational quantum circuit acting as a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the layer.
    backend : Backend or str, default 'qasm_simulator'
        Execution backend. If a string, it is resolved via Aer.get_backend.
    shots : int, default 1024
        Number of shots for measurement statistics.
    entangle : bool, default True
        Whether to add a CNOT chain after the Ry rotations.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        backend: Optional[Backend] = None,
        shots: int = 1024,
        entangle: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.entangle = entangle

        # Resolve backend
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend

        # Pre‑build the circuit template
        self._circuit = QuantumCircuit(n_qubits)
        # Parameter placeholders
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        # H on all qubits
        self._circuit.h(range(n_qubits))
        # Variational Ry rotations
        for i, t in enumerate(self.theta):
            self._circuit.ry(t, i)
        # Optional entangling layer
        if self.entangle:
            for i in range(n_qubits - 1):
                self._circuit.cx(i, i + 1)
        # Measure each qubit
        self._circuit.measure_all()

    def _expectation_z(self, counts: dict) -> np.ndarray:
        """Compute ⟨Z⟩ for each qubit from measurement counts."""
        probs = np.asarray([c / self.shots for c in counts.values()], dtype=np.float32)
        bits = np.array([list(map(int, b[::-1])) for b in counts.keys()], dtype=np.int8)
        # Z eigenvalues: +1 for 0, -1 for 1
        z_vals = 1 - 2 * bits  # shape (n_states, n_qubits)
        expectations = np.sum(z_vals * probs[:, None], axis=0)
        return expectations

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of rotation angles, one per qubit.

        Returns
        -------
        np.ndarray
            Expectation values ⟨Z⟩ for each qubit.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")

        # Bind parameters
        param_bind = {self.theta[i]: float(thetas[i]) for i in range(self.n_qubits)}
        bound_circ = self._circuit.bind_parameters(param_bind)

        # Transpile and assemble
        transpiled = transpile(bound_circ, backend=self.backend, optimization_level=3)
        qobj = assemble(transpiled, shots=self.shots, memory=True)
        job = self.backend.run(qobj)
        result: Result = job.result()
        counts = result.get_counts(transpiled)

        return self._expectation_z(counts)

    def parameter_shift_gradient(self, thetas: Iterable[float], shift: float = np.pi / 2) -> np.ndarray:
        """
        Compute gradient via the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Current parameters.
        shift : float, default π/2
            Shift value for the rule.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        grad = np.zeros(self.n_qubits, dtype=np.float32)
        for i in range(self.n_qubits):
            plus = list(thetas)
            minus = list(thetas)
            plus[i] += shift
            minus[i] -= shift
            f_plus = self.run(plus)
            f_minus = self.run(minus)
            grad[i] = (f_plus[i] - f_minus[i]) / (2 * np.sin(shift))
        return grad

__all__ = ["FCL", "get_FCL"]

def get_FCL(
    n_qubits: int = 1,
    backend: Optional[Backend] = None,
    shots: int = 1024,
    entangle: bool = True,
) -> FCL:
    """Convenience constructor mirroring the original API."""
    return FCL(
        n_qubits=n_qubits,
        backend=backend,
        shots=shots,
        entangle=entangle,
    )
