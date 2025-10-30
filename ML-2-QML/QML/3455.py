import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class HybridConvolution:
    """
    Quantum convolutional filter that encodes image patches into a parametric circuit
    and evaluates expectation values of observables.
    """
    def __init__(self, kernel_size: int = 2, backend=None,
                 shots: int = 1024, threshold: float = 0.0) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        # Simple entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self.circuit.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit for a single data patch and return the average
        probability of measuring |1>.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self._bind(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results
