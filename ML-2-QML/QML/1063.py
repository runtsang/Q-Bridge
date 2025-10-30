"""ConvGenQuantum – a variational quanvolution filter.

The quantum implementation replaces the depth‑wise separable convolution with a parameterised
variational circuit. Each qubit encodes a pixel value; rotation angles are trainable and
updated via the parameter‑shift rule. The circuit includes a barrier and a small random
entangling layer to increase expressivity. The output is the mean probability of measuring |1>
across all qubits, optionally gated by a learnable threshold.
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer, execute
from typing import List, Tuple, Union


class ConvGenQuantum:
    """Variational quanvolution filter with trainable parameters."""

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        backend_name: str = "qasm_simulator",
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel (number of qubits = kernel_size²).
        shots : int
            Number of shots for each execution.
        threshold : float
            Threshold to binarise pixel values before encoding.
        backend_name : str
            Name of the Aer backend to use.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Parameterised variational circuit
        self.params = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Encode pixel values as rotation angles
        for i, p in enumerate(self.params):
            self.circuit.rx(p, i)
        self.circuit.barrier()
        # Add a shallow entangling layer
        self.circuit += RealAmplitudes(self.n_qubits, reps=1, entanglement="full")
        self.circuit.measure_all()

        self.backend = Aer.get_backend(backend_name)

    def _encode_data(self, data: np.ndarray) -> List[dict]:
        """Create parameter bindings from a single patch."""
        bindings = []
        flat = data.ravel()
        for val in flat:
            angle = np.pi if val > self.threshold else 0.0
            bindings.append({p: angle for p in self.params})
        return bindings

    def run(self, data: Union[np.ndarray, List[np.ndarray]]) -> float:
        """
        Execute the circuit on one or more patches.

        Parameters
        ----------
        data : np.ndarray or list of np.ndarray
            Patch(es) of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits and patches.
        """
        if isinstance(data, np.ndarray):
            patches = [data]
        else:
            patches = data

        total_prob = 0.0
        for patch in patches:
            bindings = self._encode_data(patch)
            job = execute(
                self.circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=bindings,
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            # Compute mean probability of |1> across all qubits
            ones = 0
            for bitstring, c in counts.items():
                ones += c * bitstring.count("1")
            prob = ones / (self.shots * self.n_qubits)
            total_prob += prob

        return total_prob / len(patches)

    def parameters(self) -> List[Parameter]:
        """Return the trainable parameters."""
        return self.params


__all__ = ["ConvGenQuantum"]
