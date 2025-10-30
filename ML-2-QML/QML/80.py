"""Quantum implementation of a 2‑D convolution filter.

The class ``QuantumConv`` encapsulates a variational circuit that
maps a 2‑D kernel to a single scalar output.  The circuit is
parameter‑tuned by a simple ansatz and can be executed on a simulator
or a real device.  The ``run`` method accepts a 2‑D array and returns
the average probability of measuring |1⟩ across all qubits.

The design deliberately differs from the seed by:
* Using a layered entangling ansatz instead of a random circuit.
* Providing a helper that converts a classical kernel into circuit
  parameters via a threshold.
* Returning the probability in a differentiable way by sampling
  many shots and averaging.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes

__all__ = ["QuantumConv"]

class QuantumConv:
    """Variational 2‑D convolution filter implemented with Qiskit."""

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build a parameterised ansatz using RealAmplitudes
        self.theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Map each qubit to a parameterised rotation
        for i, param in enumerate(self.theta):
            self.circuit.rx(param, i)
        # Add a few layers of entanglement
        for _ in range(2):
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
            self.circuit.barrier()
        # Measure all qubits
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def _parameter_bindings(self, data: np.ndarray) -> list[dict]:
        """Create a list of parameter bindings for each sample.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        list[dict]
            Parameter bindings mapping each θ_i to either π or 0.
        """
        flat = data.reshape(-1)
        bindings = []
        for i, val in enumerate(flat):
            bindings.append({self.theta[i]: np.pi if val > self.threshold else 0.0})
        return bindings

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on the provided data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Expected data shape {(self.kernel_size, self.kernel_size)}, got {data.shape}"
            )
        # Prepare parameter bindings
        param_binds = self._parameter_bindings(data)

        # Execute
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of |1⟩ per qubit
        total_ones = 0
        total_counts = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
            total_counts += count
        prob = total_ones / (total_counts * self.n_qubits)
        return prob

    @staticmethod
    def gaussian_kernel(kernel_size: int, sigma: float = 1.0) -> np.ndarray:
        """Generate a 2‑D Gaussian kernel.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel.
        sigma : float, default 1.0
            Standard deviation of the Gaussian.

        Returns
        -------
        np.ndarray
            2‑D Gaussian kernel of shape (kernel_size, kernel_size).
        """
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return kernel
