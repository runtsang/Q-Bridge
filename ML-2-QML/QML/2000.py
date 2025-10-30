import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


class ConvEnhanced:
    """
    Pure quantum convolution filter.
    Uses a variational quantum circuit with a tunable readout schedule.
    """
    def __init__(
        self,
        kernel_size: int = 3,
        backend=None,
        shots: int = 512,
        threshold: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Parameters for RX and RZ on each qubit
        self.theta_rx = [Parameter(f"theta_rx_{i}") for i in range(self.n_qubits)]
        self.theta_rz = [Parameter(f"theta_rz_{i}") for i in range(self.n_qubits)]

        self.circuit = self._build_vqc()

    def _build_vqc(self):
        """
        Build a variational circuit with alternating RX-RZ layers and
        nearest‑neighbour entanglement.
        """
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.theta_rx[i], i)
            qc.rz(self.theta_rz[i], i)
        # Entanglement pattern
        for i in range(0, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, self.n_qubits - 1, 2):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, data):
        """
        Run the quantum filter on a 2D patch.

        Parameters
        ----------
        data : array‑like, shape (kernel_size, kernel_size)
            The image patch to filter.

        Returns
        -------
        float
            The average probability of measuring |1> across all qubits.
        """
        patch = np.array(data, dtype=np.float32).reshape(-1)
        param_bind = {}
        for i, val in enumerate(patch):
            param_bind[self.theta_rx[i]] = np.pi if val > self.threshold else 0.0
            param_bind[self.theta_rz[i]] = 0.0  # fixed for simplicity

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        ones = 0
        for bitstring, count in counts.items():
            ones += bitstring.count("1") * count
        prob = ones / (self.shots * self.n_qubits)
        return prob

    def update_threshold(self, new_threshold: float):
        """Update the binarisation threshold."""
        self.threshold = new_threshold
