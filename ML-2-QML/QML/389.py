"""ConvEnhanced: quantum variational filter for 2‑D data.

This module defines a parameterized quantum circuit that acts as a filter
for a kernel-sized patch.  The circuit can be trained by setting the
parameter values externally; the ``run`` method evaluates the expectation
value of the Z‑operator on each qubit and returns the average probability
of measuring |1>.  A threshold can be applied to binarise the input
values before parameter binding.

The module keeps a ``Conv`` factory for compatibility with the original
API.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, List

class ConvEnhanced:
    """Variational quantum filter for kernel-sized patches.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.  The number of qubits is ``kernel_size**2``.
    backend : qiskit.providers.Backend, optional
        Quantum backend to execute the circuit.  Defaults to Aer
        ``qasm_simulator``.
    shots : int, default 1024
        Number of shots per execution.
    threshold : float, default 0.0
        Threshold applied to input data before binding parameters.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameters for data encoding and trainable variational part
        self.data_param = [Parameter(f"x_{i}") for i in range(self.n_qubits)]
        self.train_theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]

        # Build the base circuit with symbolic parameters
        self.circuit = QuantumCircuit(self.n_qubits)
        # Data encoding via RX rotations
        for i in range(self.n_qubits):
            self.circuit.rx(self.data_param[i], i)
        # Variational layer with trainable parameters
        for i in range(self.n_qubits):
            self.circuit.rx(self.train_theta[i], i)
        # Simple entangling block
        self.circuit.h(range(self.n_qubits))
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

        # Store current trainable parameters (default to zeros)
        self.train_params: List[float] = [0.0] * self.n_qubits

    def set_parameters(self, params: Iterable[float]) -> None:
        """Set the trainable parameters of the circuit.

        Parameters
        ----------
        params : iterable of float
            Length must equal ``n_qubits``.
        """
        params = list(params)
        if len(params)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {len(params)}."
            )
        self.train_params = params

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single kernel patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)`` containing
            classical values in the range [0, 1].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Data shape must be {(self.kernel_size, self.kernel_size)}, "
                f"got {data.shape}."
            )
        # Flatten and binarise using threshold
        flat = data.flatten()
        bind_dict = {
            self.data_param[i]: np.pi if flat[i] > self.threshold else 0
            for i in range(self.n_qubits)
        }
        # Include trainable parameters
        bind_dict.update({
            self.train_theta[i]: self.train_params[i]
            for i in range(self.n_qubits)
        })
        bound_circ = self.circuit.bind_parameters(bind_dict)

        job = execute(
            bound_circ,
            backend=self.backend,
            shots=self.shots,
        )
        result = job.result()
        counts = result.get_counts(bound_circ)

        # Compute probability of measuring |1> on each qubit
        total = 0.0
        for bitstring, cnt in counts.items():
            ones = sum(int(b) for b in bitstring)
            total += ones * cnt

        prob = total / (self.shots * self.n_qubits)
        return prob

def Conv() -> ConvEnhanced:
    """Factory for backward compatibility."""
    return ConvEnhanced()

__all__ = ["ConvEnhanced", "Conv"]
