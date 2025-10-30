"""Quantum implementation of a hybrid convolutional layer with a classical read‑out."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


class HybridConvLayer:
    """
    Quantum counterpart of HybridConvLayer.  Image patches are encoded into
    qubit rotations, a variational ansatz is applied, and measurement
    statistics are aggregated.  The resulting probabilities are linearly
    combined to produce a scalar output.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (determines number of qubits).
    threshold : float, default 0.0
        Pixel value threshold used to set rotation angles.
    shots : int, default 100
        Number of shots for each run.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm_simulator.
    fc_weights : np.ndarray, optional
        Classical weights applied to measurement probabilities.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 100,
        backend=None,
        fc_weights: np.ndarray | None = None,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.fc_weights = fc_weights if fc_weights is not None else np.ones(self.n_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a parameterised circuit that mimics convolution."""
        qc = QuantumCircuit(self.n_qubits)
        thetas = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        # Encode pixel values into rotations
        for i, theta in enumerate(thetas):
            qc.rx(theta, i)
        qc.barrier()
        # Variational ansatz: two layers of Ry + linear CNOT chain
        for _ in range(2):
            for i, theta in enumerate(thetas):
                qc.ry(theta, i)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single image patch and return a scalar.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output derived from measurement probabilities.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        param_bind = {
            f"theta{i}": np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data_flat)
        }
        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self._circuit)

        # Compute probability of measuring |1> for each qubit
        probs = np.zeros(self.n_qubits)
        for state, count in result.items():
            for i, bit in enumerate(reversed(state)):
                probs[i] += int(bit) * count
        probs /= self.shots

        # Classical fully connected read‑out
        weighted = probs * self.fc_weights
        return weighted.mean()


__all__ = ["HybridConvLayer"]
