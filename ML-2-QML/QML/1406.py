"""Quantum‑enhanced convolutional filter using a parameterised circuit and a classical read‑out."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter


class ConvQuantum:
    """
    A drop‑in quantum version of ConvEnhanced that pre‑processes the data
    with a parameterised circuit.  The circuit consists of a single
    layer of RX rotations (one per pixel), a small entangling pattern,
    and a measurement of all qubits.  The rotation angles are
    bound at runtime based on the pixel intensity relative to a
    threshold.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter; determines the number of qubits.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit.  If None, the Aer simulator
        is used.
    shots : int, default 1024
        Number of shots for the simulation.
    threshold : float, default 0.5
        Pixel intensity threshold used to decide the rotation angle.
        Pixels above the threshold receive a π rotation, otherwise 0.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        *,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Build the parameterised circuit
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        # Simple entangling pattern: a ring of CNOTs
        for i in range(self.n_qubits):
            self.circuit.cx(i, (i + 1) % self.n_qubits)

        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

        self.backend = backend or AerSimulator()
        self.transpiled = transpile(self.circuit, self.backend)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a 2‑D array of pixel intensities.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = np.reshape(data, (self.n_qubits,))
        param_binds = {}
        for i, val in enumerate(flat):
            param_binds[self.theta[i]] = np.pi if val > self.threshold else 0.0

        bound_qc = self.transpiled.bind_parameters(param_binds)
        qobj = assemble(bound_qc, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts(bound_qc)

        # Compute average |1> probability
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq

        avg_prob = total_ones / (self.shots * self.n_qubits)
        return avg_prob

    def set_threshold(self, new_threshold: float):
        """Adjust the intensity threshold used for rotation binding."""
        self.threshold = new_threshold


__all__ = ["ConvQuantum"]
