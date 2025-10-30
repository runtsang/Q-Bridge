import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_machine_learning.utils import algorithm_globals
from typing import Iterable, Tuple, Callable

# ------------------------------------------------------------------
# Quantum autoencoder with swap‑test and domain‑wall support
# ------------------------------------------------------------------
class QuantumAutoencoderGen118:
    def __init__(self,
                 num_latent: int,
                 num_trash: int,
                 reps: int = 5,
                 seed: int | None = None):
        algorithm_globals.random_seed = seed or 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        total = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode first part with a variational ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()

        # Swap test to compare trash qubits
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def evaluate(self, params: Iterable[float]) -> complex:
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)))
        state = Statevector.from_instruction(bound)
        return state.expectation_value(SparsePauliOp("Z"))

    def sample(self, params: Iterable[float], shots: int = 1024) -> np.ndarray:
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)))
        result = self.sampler.run(bound, shots=shots)
        return result.quasi_dists[0].values()

def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    for i in range(start, end):
        circuit.x(i)
    return circuit

# ------------------------------------------------------------------
# Quantum classifier ansatz
# ------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit,
                                                                    Iterable[ParameterVector],
                                                                    Iterable[ParameterVector],
                                                                    list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    for p, q in zip(encoding, range(num_qubits)):
        qc.rx(p, q)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# ------------------------------------------------------------------
# Fast estimators for quantum circuits
# ------------------------------------------------------------------
class FastBaseEstimatorQML:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Iterable[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[SparsePauliOp],
                 parameter_sets: Iterable[Iterable[float]]) -> list[list[complex]]:
        results = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(o) for o in observables]
            results.append(row)
        return results

class FastEstimatorQML(FastBaseEstimatorQML):
    def evaluate(self,
                 observables: Iterable[SparsePauliOp],
                 parameter_sets: Iterable[Iterable[float]],
                 shots: int | None = None,
                 seed: int | None = None) -> list[list[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [rng.normal(x, max(1e-6, 1 / shots)) for x in row]
            noisy.append(noisy_row)
        return noisy

# ------------------------------------------------------------------
# Quantum regression dataset (superposition states)
# ------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

__all__ = ["QuantumAutoencoderGen118", "domain_wall",
           "build_classifier_circuit",
           "FastBaseEstimatorQML", "FastEstimatorQML",
           "generate_superposition_data"]
