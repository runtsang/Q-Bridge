"""Quantum autoencoder with fast expectation estimation."""

from __future__ import annotations

from typing import Iterable, Sequence, List
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumAutoencoderGen132:
    """Constructs a parameterized quantum autoencoder and a fast estimator."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 5,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.shots = shots
        self.seed = seed
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.estimator = self._build_estimator()

    # ----------------------------------------------------------------------- #
    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash
        qc.compose(self._build_ansatz(self.num_latent + self.num_trash), range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # ----------------------------------------------------------------------- #
    def _build_estimator(self):
        """Wraps the sampler‑QNN into a fast‑estimator style interface."""
        class _FastEstimator:
            def __init__(self, qnn: SamplerQNN, shots: int | None, seed: int | None) -> None:
                self.qnn = qnn
                self.shots = shots
                self.rng = np.random.default_rng(seed)

            def evaluate(
                self,
                observables: Iterable[BaseOperator],
                parameter_sets: Sequence[Sequence[float]],
            ) -> List[List[complex]]:
                # For the quantum autoencoder we ignore observables and return raw QNN outputs
                results: List[List[complex]] = []
                for params in parameter_sets:
                    out = self.qnn(params)
                    results.append([float(out[0]), float(out[1])])
                return results

        return _FastEstimator(self.qnn, self.shots, self.seed)

    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values using the underlying circuit."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def sample(self, parameter_set: Sequence[float], nshots: int | None = None) -> dict[str, int]:
        """Run the circuit with the given parameters and return measurement counts."""
        nshots = nshots or self.shots or 1024
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, parameter_set)), inplace=False)
        result = self.sampler.run(bound, shots=nshots).result()
        return result.get_counts()

__all__ = ["QuantumAutoencoderGen132"]
