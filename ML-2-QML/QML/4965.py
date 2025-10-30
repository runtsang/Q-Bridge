import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from typing import Iterable, List, Sequence

class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.base_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.base_circuit.rx(self.theta[i], i)
        self.base_circuit.barrier()
        self.base_circuit += random_circuit(self.n_qubits, 2)
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        circuit = self.base_circuit.copy()
        circuit.measure_all()
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class FraudGateQuantum:
    def __init__(self, backend, shots: int):
        self.backend = backend
        self.shots = shots

    def apply(self, counts):
        total = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count('1')
            total += ones * freq
        avg = total / (self.shots * len(bitstring))
        return {"gate": avg}

class QuantumSelfAttention:
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.filter = QuanvCircuit(n_qubits, self.backend, shots, threshold=0.5)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> dict:
        attention_circ = self._build_circuit(rotation_params, entangle_params)
        full_circ = self.filter.base_circuit + attention_circ
        job = execute(full_circ, self.backend, shots=self.shots)
        return job.result().get_counts(full_circ)

class FastBaseEstimator:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

class SelfAttention:
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024, use_gate: bool = False):
        self.attention = QuantumSelfAttention(n_qubits, backend, shots)
        self.gate = FraudGateQuantum(backend or Aer.get_backend("qasm_simulator"), shots) if use_gate else None

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> dict:
        counts = self.attention.run(rotation_params, entangle_params)
        if self.gate:
            return self.gate.apply(counts)
        return counts

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.attention.filter.base_circuit)
        return estimator.evaluate(observables, parameter_sets)

__all__ = ["SelfAttention"]
