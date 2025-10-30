import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel

class ConvFilter:
    """
    Variational quantum filter for 2‑D data patches.
    Constructs a parameterised circuit per patch, executes it on a
    chosen backend, and returns the expectation value of the Z‑observable
    (average probability of measuring |1>).
    """
    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
        noise_model=None,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.noise_model = noise_model

        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(theta):
            qc.ry(t, i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def _bind_parameters(self, data):
        bind = {}
        for i, val in enumerate(data):
            bind[self.circuit.parameters[i]] = np.pi if val > self.threshold else 0.0
        return bind

    def run(self, data):
        flat = data.reshape(-1)
        param_binds = [self._bind_parameters(flat)]
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
            noise_model=self.noise_model,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_counts = sum(counts.values())
        ones = 0
        for bitstring, cnt in counts.items():
            ones += cnt * bitstring.count("1")
        return ones / (total_counts * self.n_qubits)

__all__ = ["ConvFilter"]
