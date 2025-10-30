"""Quantum implementation of hybrid sampler using Qiskit.

Class: HybridSamplerQNN
Features:
- RealAmplitudes ansatz for data encoding.
- Swap test ancilla for similarity measurement.
- SamplerQNN wrapper to produce probabilities.
- QuantumKernel for kernel matrices.
- FastBaseEstimator for expectation evaluation with optional shot noise.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.quantum_info import Statevector
import numpy as np
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[Statevector], complex]

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[Statevector],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridSamplerQNN:
    """Quantum hybrid sampler using a variational circuit and a swap test."""
    def __init__(self,
                 num_qubits: int = 4,
                 reps: int = 3) -> None:
        self.num_qubits = num_qubits
        self.reps = reps

        # Data encoding ansatz
        self.ansatz = RealAmplitudes(num_qubits, reps=reps)

        # Build full circuit with ancilla for swap test
        qr = QuantumRegister(num_qubits + 1, name="q")
        cr = ClassicalRegister(1, name="c")
        self.circuit = QuantumCircuit(qr, cr)

        # Append ansatz
        self.circuit.append(self.ansatz.to_instruction(), qr[:num_qubits])

        # Ancilla qubit index
        ancilla = num_qubits
        # Swap test
        self.circuit.h(qr[ancilla])
        for i in range(num_qubits):
            self.circuit.cswap(qr[ancilla], qr[i], qr[i])  # placeholder swap
        self.circuit.h(qr[ancilla])
        self.circuit.measure(qr[ancilla], cr[0])

        # SamplerQNN wrapper
        self.sampler = QiskitSampler()
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.ansatz.parameters,
                              weight_params=[],
                              sampler=self.sampler)

        # Quantum kernel
        self.kernel = QuantumKernel(circuit=self.ansatz,
                                   quantum_instance=self.sampler)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum sampler and return probability distribution."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        probs = self.qnn.predict(inputs)
        return probs

    def compute_kernel(self,
                       a: Sequence[np.ndarray],
                       b: Sequence[np.ndarray]) -> np.ndarray:
        """Return Gram matrix between a and b using the quantum kernel."""
        return np.array([[self.kernel.evaluate(x, y) for y in b] for x in a])

    def estimator(self,
                  observables: Iterable[Statevector],
                  parameter_sets: Sequence[Sequence[float]],
                  *,
                  shots: int | None = None,
                  seed: int | None = None) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        results = estimator.evaluate(observables, parameter_sets)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [complex(rng.normal(c.real, max(1e-6, 1 / shots)),
                                 rng.normal(c.imag, max(1e-6, 1 / shots)))
                         for c in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridSamplerQNN"]
