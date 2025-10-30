import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from typing import Iterable, List, Sequence

class FastHybridEstimator:
    """
    Hybrid estimator for parameterised quantum circuits with optional shot‑noise.
    """
    def __init__(self, circuit: QuantumCircuit, *, shots: int | None = None, seed: int | None = None):
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    self.rng.normal(np.real(mean), max(1e-6, 1 / self.shots))
                    + 1j * self.rng.normal(np.imag(mean), max(1e-6, 1 / self.shots))
                )
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    @classmethod
    def FCL(cls):
        """
        Return a quantum circuit that implements a simple fully‑connected layer.
        """
        class QuantumCircuitWrapper:
            def __init__(self, n_qubits: int, backend, shots: int):
                self._circuit = qiskit.QuantumCircuit(n_qubits)
                self.theta = qiskit.circuit.Parameter("theta")
                self._circuit.h(range(n_qubits))
                self._circuit.barrier()
                self._circuit.ry(self.theta, range(n_qubits))
                self._circuit.measure_all()
                self.backend = backend
                self.shots = shots

            def run(self, thetas):
                job = qiskit.execute(
                    self._circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[{self.theta: theta} for theta in thetas],
                )
                result = job.result().get_counts(self._circuit)
                counts = np.array(list(result.values()))
                states = np.array(list(result.keys())).astype(float)
                probabilities = counts / self.shots
                expectation = np.sum(states * probabilities)
                return np.array([expectation])
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        circuit = QuantumCircuitWrapper(1, simulator, 100)
        return circuit

    @classmethod
    def Conv(cls, kernel_size: int = 2, threshold: float = 127):
        """
        Return a quantum circuit that emulates a quanvolution filter.
        """
        class QuanvCircuit:
            def __init__(self, kernel_size, backend, shots, threshold):
                self.n_qubits = kernel_size ** 2
                self._circuit = qiskit.QuantumCircuit(self.n_qubits)
                self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
                for i in range(self.n_qubits):
                    self._circuit.rx(self.theta[i], i)
                self._circuit.barrier()
                self._circuit += random_circuit(self.n_qubits, 2)
                self._circuit.measure_all()
                self.backend = backend
                self.shots = shots
                self.threshold = threshold

            def run(self, data):
                data = np.reshape(data, (1, self.n_qubits))
                param_binds = []
                for dat in data:
                    bind = {}
                    for i, val in enumerate(dat):
                        bind[self.theta[i]] = np.pi if val > self.threshold else 0
                    param_binds.append(bind)
                job = qiskit.execute(
                    self._circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=param_binds,
                )
                result = job.result().get_counts(self._circuit)
                counts = 0
                for key, val in result.items():
                    ones = sum(int(bit) for bit in key)
                    counts += ones * val
                return counts / (self.shots * self.n_qubits)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        circuit = QuanvCircuit(kernel_size, backend, shots=100, threshold=threshold)
        return circuit

__all__ = ["FastHybridEstimator"]
