"""ConvGen155: quantum filter for hybrid neural networks.

The class implements a parameterised 2‑qubit (for 2×2) filter that can be
used as a drop‑in replacement for the classical Conv.  The circuit is
constructed once and reused; the run() method binds the data to the
parameterised RX gates and returns the per‑qubit probability of measuring
|1>.  The output can be interpreted as a probability map that can be
plugged into a hybrid loss.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Tuple

__all__ = ["ConvGen155"]


class ConvGen155:
    """
    Quantum version of ConvGen155.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter (currently limited to 2×2).
    threshold : float, default 0.0
        Threshold for mapping classical pixel values to rotation angles.
    shots : int, default 1024
        Number of shots per execution.
    backend_name : str, default "qasm_simulator"
        Backend to use for execution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 1024,
        backend_name: str = "qasm_simulator",
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend(backend_name)

        # Parameterised circuit
        self.theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)

        # RX gates controlled by parameters
        for i, param in enumerate(self.theta):
            self.circuit.rx(param, i)

        # Add a small entangling layer for expressivity
        for i in range(self.n_qubits - 1):
            self.circuit.cz(i, i + 1)
        self.circuit.barrier()

        # Measurement
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        np.ndarray
            1‑D array of per‑qubit probabilities of measuring |1>.
        """
        data_flat = np.asarray(data).reshape(self.n_qubits)

        # Bind parameters according to threshold
        param_bind = {}
        for i, val in enumerate(data_flat):
            angle = np.pi if val > self.threshold else 0.0
            param_bind[self.theta[i]] = angle

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute probability of |1> for each qubit
        probs = np.zeros(self.n_qubits, dtype=float)
        for bitstring, count in counts.items():
            for i, bit in enumerate(bitstring[::-1]):  # qiskit uses little‑endian
                if bit == "1":
                    probs[i] += count
        probs /= self.shots

        return probs

    def to_probability_map(self, probs: np.ndarray) -> np.ndarray:
        """
        Reshape the flat probability vector into a 2‑D map matching the
        original kernel shape.

        Parameters
        ----------
        probs : np.ndarray
            1‑D array of per‑qubit probabilities.

        Returns
        -------
        np.ndarray
            2‑D array of shape (kernel_size, kernel_size).
        """
        return probs.reshape(self.kernel_size, self.kernel_size)
