import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class HybridFCQuanvolution:
    """
    Quantum implementation of the hybrid architecture.
    The circuit encodes a 2×2 image patch into 10 qubits, applies a random
    layer, and a variational layer with 10 parameters.  The expectation of
    each qubit approximates a log‑likelihood for one of 10 classes.
    """
    def __init__(self, backend=None, shots=1000, var_params=None):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.n_qubits = 10
        # Initialize variational parameters
        self.var_params = var_params or np.random.uniform(0, 2*np.pi, self.n_qubits)

    def _make_circuit(self, thetas: list[float]):
        # thetas: list of 4 values (one per patch) encoded on first 4 qubits
        qc = QuantumCircuit(self.n_qubits)
        # Encode input patch
        for i, theta in enumerate(thetas[:4]):
            qc.ry(theta, i)
        # Random layer: random single‑qubit rotations and CNOTs
        for i in range(self.n_qubits):
            qc.rx(np.random.uniform(0, 2*np.pi), i)
        for i in range(0, self.n_qubits-1, 2):
            qc.cx(i, i+1)
        # Variational layer
        for i, param in enumerate(self.var_params):
            qc.ry(param, i)
        # Measurement
        qc.measure_all()
        return qc

    def run(self, thetas: list[float]) -> np.ndarray:
        qc = self._make_circuit(thetas)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        # Convert counts to expectation values per qubit
        expectations = np.zeros(self.n_qubits)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            prob = count / total
            # bitstring is in order of qubits: q_{n-1}... q_0
            for i, bit in enumerate(reversed(bitstring)):
                expectations[i] += (1 - 2*int(bit)) * prob
        return expectations.reshape(1, -1)  # shape (1,10)
