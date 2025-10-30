import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence, List

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
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridAttentionAutoencoder:
    def __init__(self, n_qubits: int, n_latent: int, n_trash: int):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_trash
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(1, "c")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        # Define rotation parameters
        self.rotation_params = [Parameter(f"rot_{i}_{ax}") for i in range(self.n_qubits) for ax in ("x","y","z")]
        # Define entanglement parameters
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(self.n_qubits-1)]
        # Apply rotations
        idx = 0
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[idx], i); idx+=1
            qc.ry(self.rotation_params[idx], i); idx+=1
            qc.rz(self.rotation_params[idx], i); idx+=1
        # Controlled rotations
        for i in range(self.n_qubits-1):
            qc.crx(self.entangle_params[i], i, i+1)
        # Quantum autoencoder block
        ansatz = RealAmplitudes(self.n_latent + self.n_trash, reps=5)
        qc.compose(ansatz, range(0, self.n_latent + self.n_trash), inplace=True)
        qc.barrier()
        # Swap‑test for autoencoding
        aux = self.n_latent + 2*self.n_trash
        qc.h(aux)
        for i in range(self.n_trash):
            qc.cswap(aux, self.n_latent + i, self.n_latent + self.n_trash + i)
        qc.h(aux)
        qc.measure(aux, self.cr[0])
        return qc

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        mapping = {}
        for i in range(self.n_qubits):
            mapping[self.rotation_params[3*i]] = rotation_params[3*i]
            mapping[self.rotation_params[3*i+1]] = rotation_params[3*i+1]
            mapping[self.rotation_params[3*i+2]] = rotation_params[3*i+2]
        for i in range(self.n_qubits-1):
            mapping[self.entangle_params[i]] = entangle_params[i]
        bound = self.circuit.assign_parameters(mapping, inplace=False)
        job = execute(bound, backend, shots=shots)
        return job.result().get_counts(bound)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        est = FastBaseEstimator(self.circuit)
        return est.evaluate(observables, parameter_sets)

__all__ = ["HybridAttentionAutoencoder", "FastBaseEstimator"]
