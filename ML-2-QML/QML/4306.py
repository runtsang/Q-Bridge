"""Quantum autoencoder module with Qiskit, leveraging FastBaseEstimator and dataset utilities."""

from __future__ import annotations

import numpy as np
import torch

from typing import Iterable, List

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as SamplerPrimitive
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
# Dataset utilities (mirroring the classical counterpart)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    return states, labels

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset exposing the synthetic data in a PyTorch-friendly format."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Fast estimator quantum
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> list[list[complex]]:
        observables = list(observables)
        results: list[list[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum autoencoder construction
# --------------------------------------------------------------------------- #
def quantum_autoencoder(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_trash + num_latent + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])
    return circuit

def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    for i in range(int(b / 2), int(b)):
        circuit.x(i)
    return circuit

def autoencoder_with_domain_wall(num_latent: int, num_trash: int) -> QuantumCircuit:
    ae_circ = quantum_autoencoder(num_latent, num_trash)
    dw_circ = domain_wall(QuantumCircuit(num_latent + 2 * num_trash + 1), 0, num_latent + 2 * num_trash + 1)
    full = dw_circ.compose(ae_circ)
    return full

# --------------------------------------------------------------------------- #
# Sampler QNN wrapper
# --------------------------------------------------------------------------- #
def AutoencoderQNN(num_latent: int, num_trash: int) -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = SamplerPrimitive()
    qc = autoencoder_with_domain_wall(num_latent, num_trash)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Quantum autoencoder wrapper class
# --------------------------------------------------------------------------- #
class Autoencoder:
    """Wrapper around a Qiskit SamplerQNN that behaves like a neural network."""
    def __init__(self, num_latent: int, num_trash: int):
        self.qnn = AutoencoderQNN(num_latent, num_trash)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return predictions from the underlying QNN."""
        return self.qnn.predict(data)

    def set_weights(self, params: np.ndarray) -> None:
        """Update the QNN weights."""
        self.qnn.set_weights(params)

    @property
    def trainable_params(self) -> list[float]:
        return list(self.qnn.trainable_params)

# --------------------------------------------------------------------------- #
# Training loop for quantum autoencoder
# --------------------------------------------------------------------------- #
def train_quantum_autoencoder(
    model: Autoencoder,
    data: np.ndarray,
    *,
    epochs: int = 50,
    shots: int = 1024,
    initial_params: np.ndarray | None = None,
) -> list[float]:
    if initial_params is None:
        initial_params = np.random.uniform(0, 2 * np.pi, size=len(model.trainable_params))
    optimizer = COBYLA()
    history: list[float] = []

    for epoch in range(epochs):
        def loss_fn(params):
            model.set_weights(params)
            outputs = model.predict(data)
            # Assume the first output channel encodes the reconstruction
            recon = outputs[:, 0]
            target = data[:, 0]
            return np.mean((recon - target) ** 2)

        opt_result = optimizer.optimize(num_vars=len(initial_params), objective_function=loss_fn, initial_point=initial_params)
        initial_params = opt_result[0]
        history.append(opt_result[1])
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderQNN",
    "generate_superposition_data",
    "QuantumRegressionDataset",
    "FastBaseEstimator",
    "train_quantum_autoencoder",
]
