import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import SPSA

class QuantumClassifier:
    """Quantum variational classifier with adaptive entanglement and multi‑observable readout."""
    def __init__(self, num_qubits: int, depth: int = 3, entanglement: str = 'full'):
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(num_qubits, depth)

    def build_classifier_circuit(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            if self.entanglement == 'full':
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qc.cz(i, j)
            elif self.entanglement == 'nearest':
                for i in range(num_qubits - 1):
                    qc.cz(i, i + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        observables.append(SparsePauliOp("Z" * num_qubits))
        return qc, list(encoding), list(weights), observables

    def get_expectation(self, params: np.ndarray, data_point: np.ndarray) -> np.ndarray:
        bound = self.circuit.bind_parameters({**{p: v for p, v in zip(self.encoding, data_point)},
                                             **{w: v for w, v in zip(self.weights, params)}})
        backend = Aer.get_backend("aer_simulator_statevector")
        result = execute(bound, backend, shots=1024).result()
        state = result.get_statevector(bound)
        exp_vals = []
        for op in self.observables:
            exp_vals.append(np.real(op.matrix().dot(state).dot(state.conj())))
        return np.array(exp_vals)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.01):
        opt = SPSA(maxiter=epochs, initial_point=np.zeros(self.num_qubits * self.depth))
        def cost(params):
            total = 0.0
            for x, label in zip(X, y):
                exp = self.get_expectation(params, x)
                pred = 1 if exp[0] > exp[1] else 0
                total += (pred - label) ** 2
            return total / len(X)
        self.weights = opt.optimize(cost)[0]

__all__ = ["QuantumClassifier"]
