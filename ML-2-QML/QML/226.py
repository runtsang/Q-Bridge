import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class ConvolutionFilter:
    """
    Quantum filter that emulates a 2‑D convolutional kernel.

    The filter encodes the input pixels into rotation angles,
    applies a random entangling circuit, and measures the
    probability of observing |1> on each qubit.  The average
    probability is returned as the filter response.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 threshold: float = 0.5,
                 shots: int = 1024,
                 backend: qiskit.providers.Backend | None = None) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel (number of qubits = kernel_size**2).
        threshold : float
            Binary threshold applied to input pixels before encoding.
        shots : int
            Number of shots for the simulator.
        backend : qiskit.providers.Backend, optional
            Qiskit backend to execute the circuit.  If None, the
            Aer qasm simulator is used.
        """
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.n_qubits = kernel_size ** 2
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f'theta_{i}') for i in range(self.n_qubits)]
        # Encode data into Rx angles
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        # Random entangling layer (depth 2)
        for _ in range(2):
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i+1)
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i+1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values
            in the range [0, 1].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        param_binds = {f'theta_{i}': np.pi if val > self.threshold else 0
                       for i, val in enumerate(flat)}
        bound_qc = self.circuit.bind_parameters(param_binds)
        qobj = assemble(transpile(bound_qc, self.backend), shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts(bound_qc)
        total_ones = sum(int(key, 2).bit_count() * cnt for key, cnt in counts.items())
        return total_ones / (self.shots * self.n_qubits)
