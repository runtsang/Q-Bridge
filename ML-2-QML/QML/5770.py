import numpy as np
from typing import Iterable, Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN

class AutoencoderHybridQNN:
    """Variational quantum autoencoder with a swap‑test latent read‑out."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, reps: int = 5):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz for latent + trash register
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)
        qc.barrier()

        # Swap‑test between latent and trash qubits
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, parameters: Sequence[float]) -> np.ndarray:
        """Return the measurement probabilities of the swap‑test ancilla."""
        result = self.sampler.run(
            self.circuit,
            parameter_binds=[dict(zip(self.circuit.parameters, parameters))],
        )
        counts = result.quasi_dists[0]
        return np.array([counts.get("0", 0), counts.get("1", 0)])

    def decode(self, latent: Sequence[float]) -> np.ndarray:
        """Placeholder for a classical decoder; to be implemented externally."""
        raise NotImplementedError("Quantum decoder not defined; use a classical decoder.")

    def forward(self, parameters: Sequence[float]) -> np.ndarray:
        return self.decode(self.encode(parameters))

class FastEstimatorQuantum:
    """Fast evaluation of expectation values for a parameterised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        obs = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(o) for o in obs])
        return results

__all__ = ["AutoencoderHybridQNN", "FastEstimatorQuantum"]
