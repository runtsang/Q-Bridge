"""Conv: variational quantum convolution filter.

This class implements a parameterised quantum circuit that acts as
a feature map for 2D image patches.  The circuit is built on a
grid of qubits whose size equals the kernel area.  Each qubit
receives a data‑dependent rotation around the X‑axis; the
entanglement pattern can be extended to include a trainable
ansatz.  The `run` method evaluates the average probability of
measuring |1> across all qubits, providing a scalar output that
matches the behaviour of the original seed.

The class is drop‑in compatible with the classical ConvFilter:
`Conv()` returns an instance that can be used as a filter in the
same way.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute


class Conv:
    """
    Parameters
    ----------
    kernel_size : int
        Size of the square kernel; the circuit will have
        kernel_size**2 qubits.
    threshold : float, default 0.0
        Data threshold used to decide whether a pixel
        triggers a π rotation.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm_simulator.
    shots : int, default 1024
        Number of shots per circuit execution.
    entangle : bool, default True
        If True, adds a layer of CNOTs between neighbouring qubits
        to introduce correlations.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        entangle: bool = True,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.entangle = entangle

        # Parameterised circuit
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Data‑dependent RX gates
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        if self.entangle:
            # Simple linear entanglement chain
            for i in range(self.n_qubits - 1):
                self._circuit.cx(i, i + 1)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the circuit on a 2D array `data` of shape
        (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten data and reshape to match qubit ordering
        flat = np.reshape(data, (1, self.n_qubits))

        binds = []
        for vec in flat:
            bind = {theta: (np.pi if val > self.threshold else 0) for theta, val in zip(self.theta, vec)}
            binds.append(bind)

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute expectation: mean of |1> populations
        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(b) for b in bitstring[::-1])  # LSB first
            total_ones += ones * count

        return total_ones / (self.shots * self.n_qubits)

def ConvFactory(kernel_size: int = 2, threshold: float = 0.0) -> Conv:
    """Convenience factory that mimics the original seed's signature."""
    return Conv(kernel_size=kernel_size, threshold=threshold)

__all__ = ["Conv", "ConvFactory"]
