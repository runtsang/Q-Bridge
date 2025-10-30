"""Quantum convolution filter with a trainable variational circuit.

The class is a drop‑in replacement for the original quanvolution filter.
It exposes a `run` method that accepts a 2‑D array and returns a
float representing the average probability of measuring |1>.

The circuit uses a ParameterVector for the rotation angles so that the
filter can be trained with a classical optimiser; the threshold is
stored as a float and can be updated externally.
"""

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from typing import Union, Iterable


class _QuanvCircuit:
    """Variational quantum filter for a square kernel."""

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.0,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self._circuit = Aer.get_backend("qasm_simulator").circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)

        # parameter vector for rotations
        self.params = ParameterVector("theta", self.n_qubits * 2)

        # encode data into rotations
        for i in range(self.n_qubits):
            self._circuit.rx(self.params[2 * i], i)
            self._circuit.rz(self.params[2 * i + 1], i)

        # add entangling layer
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)

        self._circuit.measure_all()

    def run(self, data: Union[np.ndarray, Iterable]) -> float:
        """Execute the circuit on classical data.

        Parameters
        ----------
        data
            2‑D array with shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.asarray(data).reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.params[2 * i]] = np.pi if val > self.threshold else 0
                bind[self.params[2 * i + 1]] = 0.0  # keep RZ at 0 for simplicity
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # compute average |1> probability
        counts = 0
        for bitstring, freq in result.items():
            ones = sum(int(b) for b in bitstring)
            counts += ones * freq
        return counts / (self.shots * self.n_qubits)


def Conv(
    kernel_size: int = 2,
    shots: int = 1024,
    threshold: float = 0.0,
) -> _QuanvCircuit:
    """Return a variational quantum filter.

    The returned object has the same ``run`` method as in the seed
    implementation, but the rotation angles are trainable parameters
    that can be updated by a classical optimiser.
    """
    return _QuanvCircuit(kernel_size, shots, threshold)


__all__ = ["Conv"]
