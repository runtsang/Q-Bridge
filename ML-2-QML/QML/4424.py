"""Quantum autoencoder implementation using qiskit and fast estimator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Callable

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN

# ----------------------------------------------------------------------
# Fast estimator utilities (adapted from the FastBaseEstimator pair)
# ----------------------------------------------------------------------
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

    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(complex(val.real, val.imag), max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# Quantum autoencoder
# ----------------------------------------------------------------------
@dataclass
class QuantumAutoencoderConfig:
    """Configuration for :class:`QuantumAutoencoder`."""
    num_latent: int = 3
    num_trash: int = 2
    reps: int = 5


class QuantumAutoencoder:
    """Variational quantum autoencoder built on a swap‑test based circuit."""
    def __init__(self, config: QuantumAutoencoderConfig) -> None:
        self.config = config
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.config.reps)

    def _auto_encoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(
            self._ansatz(num_latent + num_trash),
            range(0, num_latent + num_trash),
            inplace=True,
        )
        circuit.barrier()
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def _build_circuit(self) -> QuantumCircuit:
        return self._auto_encoder_circuit(self.config.num_latent, self.config.num_trash)

    def encode(self, params: List[float]) -> List[complex]:
        """Return the measurement outcome for a given parameter set."""
        return self.qnn.forward(params)

    def train(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[float]:
        """Simple gradient‑based training using COBYLA on the MSE between output and target."""
        optimizer = COBYLA()
        history: List[float] = []

        num_params = len(self.circuit.parameters)

        def loss_fn(params: Sequence[float]) -> float:
            preds = self.encode(params)
            target = data.squeeze().tolist()
            preds_tensor = torch.tensor([abs(p) for p in preds], dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return float(((preds_tensor - target_tensor) ** 2).mean().item())

        for _ in range(epochs):
            result = optimizer.optimize(
                lambda p: loss_fn(p),
                num_params,
                initial_point=np.random.rand(num_params),
            )
            params = result[0]
            history.append(loss_fn(params))
        return history

    def evaluate(
        self,
        observables: Iterable[Statevector],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        estimator_cls = FastEstimator if shots is not None else FastBaseEstimator
        estimator = estimator_cls(self.circuit)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = [
    "QuantumAutoencoder",
    "QuantumAutoencoderConfig",
    "train",
    "evaluate",
]
