import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Parameterised multi‑qubit circuit with tunable depth.
    The circuit consists of alternating layers of a Hadamard on every qubit,
    a Ry rotation with a distinct parameter for each qubit, and a chain of
    CNOT gates that entangles all qubits.  The depth controls how many
    repetitions of this block are performed.
    """
    def __init__(self,
                 n_qubits: int,
                 depth: int = 1,
                 backend: qiskit.providers.BaseBackend = None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or AerSimulator()
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QC(self.n_qubits)
        self.param_names = []
        for d in range(self.depth):
            self.circuit.h(range(self.n_qubits))
            self.circuit.barrier()
            for q in range(self.n_qubits):
                pname = f"theta_{d}_{q}"
                param = qiskit.circuit.Parameter(pname)
                self.param_names.append(param)
                self.circuit.ry(param, q)
            if self.n_qubits > 1:
                for i in range(self.n_qubits - 1):
                    self.circuit.cx(i, i + 1)
            self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of angle sets.  `angles` must
        be of shape (batch, n_qubits * depth) where each row contains
        the parameters for the corresponding circuit run.
        """
        if angles.ndim == 1:
            angles = angles.reshape(1, -1)
        compiled = transpile(self.circuit, self.backend)
        param_bindings = []
        for theta_set in angles:
            params = {}
            for d in range(self.depth):
                for q in range(self.n_qubits):
                    idx = d * self.n_qubits + q
                    pname = f"theta_{d}_{q}"
                    params[pname] = theta_set[idx]
            param_bindings.append(params)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=param_bindings)
        job = self.backend.run(qobj)
        result = job.result()
        expectations = []
        for idx in range(angles.shape[0]):
            counts = result.get_counts(self.circuit, key=idx)
            probs = np.array(list(counts.values())) / self.shots
            bitstrings = np.array(list(counts.keys()))
            # Expectation of Z on qubit 0: +1 for |0⟩, -1 for |1⟩
            z_vals = np.where(bitstrings[:, 0] == '0', 1, -1)
            expectation = np.sum(z_vals * probs)
            expectations.append(expectation)
        return np.array(expectations)

__all__ = ["QuantumCircuitWrapper"]
