"""Hybrid quantum‑classical autoencoder.

The implementation uses:
* a 2‑D quantum convolution circuit (QuanvCircuit) to extract features
* a variational autoencoder circuit that performs encoding and a swap‑test based
  decoding (auto_encoder_circuit)
* a SamplerQNN to interpret measurement outcomes
* FastBaseEstimator to evaluate expectation values with optional shot noise
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Callable, Iterable, List, Sequence

# --------------------------------------------------------------------------- #
# Quantum convolution filter
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum convolution filter emulating a 2‑D quanvolution."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [ParameterVector(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += RealAmplitudes(self.n_qubits, reps=2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Return average probability of |1⟩ across qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit, self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Variational autoencoder circuit
# --------------------------------------------------------------------------- #
def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build a variational autoencoder with a swap‑test decoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    # Encode part
    circuit.compose(RealAmplitudes(num_latent + num_trash, reps=5),
                    range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    # Swap‑test decoder
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

# --------------------------------------------------------------------------- #
# Fast estimator for expectation values
# --------------------------------------------------------------------------- #
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
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Hybrid quantum autoencoder
# --------------------------------------------------------------------------- #
class QuantumAutoencoderGen227:
    """Hybrid quantum autoencoder combining a quantum convolution filter,
    a variational autoencoder circuit, and a sampler‑based neural network.
    """
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 shots: int = 1024):
        algorithm_globals.random_seed = 42
        self.backend = Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = auto_encoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=[],
                              weight_params=self.circuit.parameters,
                              sampler=self.sampler)
        self.estimator = FastBaseEstimator(self.circuit)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return a latent representation using the quantum circuit."""
        # For simplicity, use the statevector expectation of the first qubit
        sv = Statevector.from_instruction(self.circuit)
        return np.real(sv.data)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vector back to data space via the sampler."""
        probs = self.sampler.run(self.circuit, parameter_binds=[{}])
        return np.array(probs)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Delegate to FastBaseEstimator."""
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["QuantumAutoencoderGen227", "QuanvCircuit", "auto_encoder_circuit", "FastBaseEstimator"]
