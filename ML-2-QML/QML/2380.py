import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

class HybridClassifierQML:
    """
    Quantum counterpart of HybridClassifier.  It first runs a small
    data‑uploading quanvolution circuit to extract a feature vector
    and then applies a layered ansatz with explicit encoding and
    variational parameters.  The public API mirrors the classical
    build_classifier_circuit so that hybrid experiments can be
    written once and executed on either backend.
    """
    def __init__(self, num_qubits: int, depth: int, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        self.backend = Aer.get_backend("qasm_simulator")
        self.conv = self._build_quanv_circuit(kernel_size, self.backend, shots, threshold)
        self.classifier, self.encoding, self.weights, self.observables = self.build_classifier_circuit(num_qubits, depth)

    def _build_quanv_circuit(self, kernel_size: int, backend, shots: int, threshold: float):
        n_qubits = kernel_size ** 2
        circuit = QuantumCircuit(n_qubits)
        theta = [circuit.parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circuit.rx(theta[i], i)
        circuit.barrier()
        circuit += random_circuit(n_qubits, 2)
        circuit.measure_all()
        return {
            "circuit": circuit,
            "theta": theta,
            "backend": backend,
            "shots": shots,
            "threshold": threshold,
            "n_qubits": n_qubits
        }

    def run_quanv(self, data: np.ndarray) -> float:
        """
        Execute the quanvolution circuit on a 2‑D kernel and return
        the average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.conv["n_qubits"]))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.conv["theta"][i]] = np.pi if val > self.conv["threshold"] else 0
            param_binds.append(bind)
        job = execute(self.conv["circuit"], self.conv["backend"],
                      shots=self.conv["shots"], parameter_binds=param_binds)
        result = job.result().get_counts(self.conv["circuit"])
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.conv["shots"] * self.conv["n_qubits"])

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

__all__ = ["HybridClassifierQML"]
