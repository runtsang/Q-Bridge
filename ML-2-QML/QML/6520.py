import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

class QCNNHybridKernelQuantum:
    """
    Quantum kernel implementation using Qiskit. The kernel is defined as the squared
    overlap between two quantum states generated from classical feature vectors.
    Each feature vector is encoded into a circuit of Ry rotations on n_wires qubits.
    The kernel matrix is computed by evaluating the inner product of the resulting
    statevectors.
    """

    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.backend = Aer.get_backend('statevector_simulator')

    def encode_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """Return a circuit that encodes the given feature vector."""
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(features):
            qc.ry(val, i)
        return qc

    def statevector(self, features: np.ndarray) -> Statevector:
        """Compute the statevector of the circuit encoding the given features."""
        qc = self.encode_circuit(features)
        result = execute(qc, self.backend).result()
        return Statevector(result.get_statevector(qc))

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two sets of feature vectors."""
        n, m = X.shape[0], Y.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            sv_x = self.statevector(X[i])
            for j in range(m):
                sv_y = self.statevector(Y[j])
                K[i, j] = np.abs(np.vdot(sv_x.data, sv_y.data)) ** 2
        return K

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the quantum kernel output for a batch of feature vectors.
        The output is a vector of kernel values between each input and a
        reference set (here we use the first batch element as reference).
        """
        ref = x[0:1]  # Use the first sample as a reference
        K = self.kernel_matrix(x, ref)
        return K.squeeze()
