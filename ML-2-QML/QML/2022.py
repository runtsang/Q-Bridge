from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """Quantum classifier using a parameter‑efficient ansatz and amplitude encoding."""
    def __init__(self, num_qubits: int, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weight_params, self.observables = self.build_classifier_circuit(num_qubits, depth)
        self.backend = Aer.get_backend("statevector_simulator")

    def build_classifier_circuit(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a tuple (circuit, encoding, weight_sizes, observables)."""
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
        return circuit, list(encoding), [len(weights)], observables

    def evaluate(self, x: np.ndarray, param_values: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for input x and parameter vector param_values. Returns expectation values."""
        param_dict = {str(p): v for p, v in zip(self.encoding_params + self.weight_params, param_values)}
        bound_circuit = self.circuit.bind_parameters(param_dict)
        result = execute(bound_circuit, self.backend, shots=1).result()
        statevector = result.get_statevector(bound_circuit)
        probs = np.abs(statevector) ** 2
        exp_vals = []
        for op in self.observables:
            eigvals = op.eigenvalues()
            exp = np.dot(probs, eigvals)
            exp_vals.append(exp)
        return np.array(exp_vals)

__all__ = ["QuantumClassifierModel"]
