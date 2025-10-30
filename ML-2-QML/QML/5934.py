import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

class QuantumFullyConnectedKernel:
    """
    Quantum implementation that mirrors the classical module.

    Features:
    * A parameterised Ry‑circuit acting as a fully‑connected layer.
    * A quantum kernel based on state‑vector overlap.
    The class exposes a `forward` method returning both parts and a
    `kernel_matrix` helper, matching the API of the classical version.
    """
    def __init__(self,
                 n_qubits: int = 1,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._theta = Parameter("θ")
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self._theta, range(n_qubits))
        self._circuit.measure_all()

    def _run_fc(self, theta: float) -> np.ndarray:
        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[{self._theta: theta}])
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / self.shots
        return np.array([np.sum(states * probs)])

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Quantum kernel via state‑vector overlap."""
        qc_x = QuantumCircuit(self.n_qubits)
        qc_x.h(range(self.n_qubits))
        qc_x.ry(x[0], range(self.n_qubits))
        qc_x.save_statevector()
        qc_y = QuantumCircuit(self.n_qubits)
        qc_y.h(range(self.n_qubits))
        qc_y.ry(y[0], range(self.n_qubits))
        qc_y.save_statevector()

        sv_x = execute(qc_x, self.backend, shots=1).result().get_statevector(qc_x)
        sv_y = execute(qc_y, self.backend, shots=1).result().get_statevector(qc_y)
        overlap = np.abs(np.vdot(sv_x, sv_y)) ** 2
        return np.array([overlap])

    def forward(self,
                x: np.ndarray,
                theta: float | None = None) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (batch, features).  Only the first feature
            is used in this toy example.
        theta : float, optional
            Parameter for the FC part.  When omitted only the kernel part
            is returned.

        Returns
        -------
        np.ndarray
            Concatenated FC output and kernel value.
        """
        fc = self._run_fc(theta) if theta is not None else np.array([0.0])
        k = self._kernel(x[0], x[0])
        return np.concatenate([fc, k])

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two sets of samples."""
        m, n = a.shape[0], b.shape[0]
        mat = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                mat[i, j] = self._kernel(a[i], b[j])[0]
        return mat

def FCL() -> QuantumFullyConnectedKernel:
    """Compatibility shim mirroring the original FCL interface."""
    return QuantumFullyConnectedKernel()

__all__ = ["QuantumFullyConnectedKernel", "FCL"]
