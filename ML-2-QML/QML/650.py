import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit

class ConvEnhanced:
    """Quantum implementation of a convolution filter.  The circuit generates a
    probability map that can be interpreted as a kernel.  The public interface
    matches the classical version: an instance exposes a run(data) method that
    accepts a 2‑D array of shape (kernel_size, kernel_size) and returns a single
    float representing the filter response.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        shots: int = 1000,
        backend=None,
        device: str = "cpu",
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.device = device

        # Build the variational circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        n = self.kernel_size ** 2
        qc = QuantumCircuit(n, n)
        # Parameterised Ry gates
        thetas = [Parameter(f"θ{i}") for i in range(n)]
        for i, th in enumerate(thetas):
            qc.ry(th, i)
        # Entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure(range(n), range(n))
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the given patch and return the average probability
        of measuring |1> across all qubits.
        """
        flat = data.flatten()
        binds = []
        for i, val in enumerate(flat):
            bind = {self.circuit.parameters[i]: np.pi if val > self.threshold else 0}
            binds.append(bind)
        # Execute
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt
        avg_prob = total_ones / (total_counts * self.kernel_size ** 2)
        return avg_prob

    def get_kernel(self) -> np.ndarray:
        """
        Run the circuit on a dummy patch of all zeros to obtain a probability map,
        which is interpreted as a convolution kernel.
        """
        dummy = np.zeros((self.kernel_size, self.kernel_size))
        probs = self.run(dummy)
        kernel = np.full((self.kernel_size, self.kernel_size), probs)
        return kernel
