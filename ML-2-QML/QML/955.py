"""Variational fully connected layer using a parameterised quantum circuit.

The circuit implements a single‑qubit variational layer that can be
used as a drop‑in replacement for the classical fully connected layer.
It provides a ``run`` method that evaluates the expectation value of
Pauli‑Z on the first qubit and a ``gradient`` method that returns the
gradient with respect to the circuit parameters using the parameter‑shift rule.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector

class FCL:
    """Variational quantum circuit representing a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (one per input feature).
    backend : qiskit.providers.Backend, optional
        Quantum backend to execute the circuit.  If ``None`` a local
        Aer qasm simulator is used.
    shots : int, default=1024
        Number of shots for expectation estimation.
    """

    def __init__(self, n_qubits: int = 1,
                 backend: qiskit.providers.Backend = None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.theta = ParameterVector("theta", length=n_qubits)
        self._circuit = QuantumCircuit(n_qubits)
        # Simple ansatz: H on all qubits, followed by RY(theta) on each qubit
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.barrier()
        # Measure all qubits
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return the expectation value of Z on qubit 0.

        Parameters
        ----------
        thetas : np.ndarray
            Array of shape ``(n_qubits,)`` containing the rotation angles.

        Returns
        -------
        np.ndarray
            Array of shape ``(1,)`` containing the expectation value.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")
        param_bind = {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
        job = execute(self._circuit, self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bind])
        result = job.result().get_counts(self._circuit)
        # Convert measurement strings to integers
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])  # binary to int
        probs = counts / self.shots
        # Expectation of Z on qubit 0: map bit 0 to +/-1
        # bits are in little‑endian order, so bit 0 is least significant
        z_vals = 1 - 2 * (states & 1)  # 0 -> +1, 1 -> -1
        expectation = np.sum(z_vals * probs)
        return np.array([expectation])

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """Compute gradient of the expectation w.r.t. each parameter using
        the parameter‑shift rule.

        Parameters
        ----------
        thetas : np.ndarray
            Current parameter values.

        Returns
        -------
        np.ndarray
            Gradient array of shape ``(n_qubits,)``.
        """
        shift = np.pi / 2
        grad = np.zeros_like(thetas, dtype=float)
        for i in range(self.n_qubits):
            thetas_plus = thetas.copy()
            thetas_minus = thetas.copy()
            thetas_plus[i] += shift
            thetas_minus[i] -= shift
            f_plus = self.run(thetas_plus)[0]
            f_minus = self.run(thetas_minus)[0]
            grad[i] = (f_plus - f_minus) / 2.0
        return grad
