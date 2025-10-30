"""Quantum hybrid autoencoder using variational circuits and SamplerQNN."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  HybridAutoencoder – quantum variant
# --------------------------------------------------------------------------- #

class HybridAutoencoder:
    """Quantum autoencoder built from a RealAmplitudes ansatz and a swap‑test encoder."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 5,
        seed: int | None = None,
    ) -> None:
        algorithm_globals.random_seed = seed or 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._identity_interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def _identity_interpret(self, x: Sequence[float]) -> Sequence[float]:
        """Pass‑through interpretation – keeps the raw probability vector."""
        return list(x)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz on latent+trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test for latent vs. trash
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def forward(self, parameters: Sequence[float]) -> Sequence[float]:
        """Run the circuit with the supplied parameters and return measurement probabilities."""
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, parameters)), inplace=False)
        result = self.sampler.run(bound, shots=1024).result()
        probs = result.get_counts(bound)
        # Convert counts to a probability vector (0/1 measurement)
        return [probs.get("0", 0) / 1024, probs.get("1", 0) / 1024]

    def evaluate(
        self,
        observables: Iterable[Sequence[float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        return FastBaseEstimator(self.circuit).evaluate(observables, parameter_sets)

    def train(
        self,
        data: Sequence[Sequence[float]],
        target: Sequence[Sequence[float]],
        *,
        maxiter: int = 200,
        tol: float = 1e-4,
    ) -> List[float]:
        """Train the circuit parameters to minimise MSE between qnn output and target."""
        def objective(params: np.ndarray) -> float:
            loss = 0.0
            for x, y in zip(data, target):
                self.qnn.set_parameters(params)
                out = self.qnn.forward(x)
                loss += np.mean((np.array(out) - np.array(y)) ** 2)
            return loss / len(data)

        opt = COBYLA(maxiter=maxiter, tol=tol)
        initial = np.random.random(len(self.circuit.parameters))
        result = opt.optimize(initial, objective)
        return result[1]  # loss history

# --------------------------------------------------------------------------- #
#  FastBaseEstimator – quantum utility
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Sequence[float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  Factory helper
# --------------------------------------------------------------------------- #

def HybridAutoencoderFactory(
    num_latent: int,
    num_trash: int,
    reps: int = 5,
    seed: int | None = None,
) -> HybridAutoencoder:
    """Convenient factory mirroring the classical factory."""
    return HybridAutoencoder(num_latent, num_trash, reps, seed)

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train",
]
