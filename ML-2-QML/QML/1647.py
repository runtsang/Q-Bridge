"""Quantum convolutional filter using a variational circuit.

The function Conv() returns an instance of :class:`ConvGen` that implements
the same interface as the classical version.  The circuit contains
parameterised rotations and a shallow entanglement pattern.  The
run() method accepts a 2‑D array of pixel values, encodes them into
rotation angles, executes the circuit on a chosen backend, and
returns the average probability of measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

__all__ = ["Conv"]


class ConvGen:
    """Quantum filter that emulates a convolutional kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (the filter will use ``kernel_size**2``
        qubits).
    shots : int
        Number of shots for each execution.
    threshold : float
        Pixel value that determines the rotation angle (0 or π).
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit on.  If None, the Aer qasm
        simulator is used.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 200,
        threshold: float = 127,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the variational circuit
        self._circuit = QuantumCircuit(self.n_qubits)
        self.params = ParameterVector("theta", self.n_qubits)

        # Data‑dependent rotations
        for i, theta in enumerate(self.params):
            self._circuit.ry(theta, i)

        # Entangling layer: a simple ring of CNOTs
        for i in range(self.n_qubits):
            self._circuit.cx(i, (i + 1) % self.n_qubits)

        self._circuit.measure_all()

    def run(self, data: np.ndarray | list | tuple) -> float:
        """Execute the quantum filter on *data*.

        Parameters
        ----------
        data
            2‑D array of shape (kernel_size, kernel_size) with integer
            pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.asarray(data).reshape(1, self.n_qubits)

        param_binds = []
        for pixel_array in data:
            bind = {}
            for i, val in enumerate(pixel_array):
                bind[self.params[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        counts = job.result().get_counts(self._circuit)

        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += freq * bitstring.count("1")

        return total_ones / (self.shots * self.n_qubits)

def Conv(*args, **kwargs) -> ConvGen:
    """Drop‑in replacement for the original Conv factory.

    Arguments are forwarded to :class:`ConvGen`.  The function returns an
    instantiated quantum filter so existing code that calls ``Conv()`` can
    switch to the quantum version without modification.
    """
    return ConvGen(*args, **kwargs)
