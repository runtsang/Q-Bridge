"""Quantum convolutional filter implemented with a variational circuit.

The ConvGen class mirrors the classical API: a Conv() factory returns an instance,
and a run() method accepts a 2‑D NumPy array and returns a scalar probability
of measuring |1> on the variational circuit.  The circuit uses parameterised RX gates
to encode pixel values, followed by a shallow entangling ansatz.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

class ConvGen:
    """
    Variational quantum filter for kernel‑sized image patches.
    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (e.g. 2 → 4 qubits).
    threshold : float, default 127.0
        Pixel intensity threshold used to binarise the input.
    backend : qiskit.providers.BaseBackend, optional
        Backend to execute the circuit; defaults to Aer qasm_simulator.
    shots : int, default 1024
        Number of shots per circuit execution.
    depth : int, default 2
        Depth of the entangling variational ansatz.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        backend=None,
        shots: int = 1024,
        depth: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.depth = depth
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a parameterised RX‑based circuit with a shallow entangling ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        # RX rotation parameters
        theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(theta):
            qc.rx(p, i)
        # Variational entangling layers
        for _ in range(self.depth):
            # Entangle neighbours
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            # Add a second layer of RX gates with the same parameters
            for i in range(self.n_qubits):
                qc.rx(theta[i], i)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the variational circuit on a single kernel patch.
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with pixel intensities.
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten data to 1‑D
        flat = np.reshape(data, (self.n_qubits,))
        # Build parameter bindings based on threshold
        param_binds = []
        for _ in range(1):  # single data instance
            bind = {}
            for i, val in enumerate(flat):
                bind[f"theta{i}"] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)
        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        # Compute average |1> probability
        total_ones = 0
        for outcome, count in result.items():
            ones = sum(int(bit) for bit in outcome)
            total_ones += ones * count
        return total_ones / (self.shots * self.n_qubits)

def Conv() -> ConvGen:
    """
    Factory that returns a default ConvGen instance, preserving the original API.
    """
    return ConvGen()

__all__ = ["Conv"]
