"""Quantum autoencoder implementation using Qiskit.

The module defines a quantum autoencoder circuit with a 224‑dimensional
latent space encoded into a small number of qubits via a feature map.
It also provides a FastEstimator for evaluating expectation values
efficiently on a statevector sampler.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np
from collections.abc import Iterable, Sequence
import warnings

# Utility: generate a feature map that encodes a 224‑dimensional vector into qubits
def _feature_map(num_qubits: int, reps: int = 2) -> ZZFeatureMap:
    return ZZFeatureMap(feature_dimension=224, reps=reps, entanglement="full")

# Quantum autoencoder circuit
def QuantumAutoencoderCircuit(latent_dim: int, num_trash: int = 2, reps: int = 3) -> QuantumCircuit:
    """Constructs a quantum autoencoder circuit.

    The circuit uses a feature map to encode the input into qubits,
    then a RealAmplitudes ansatz for the latent part, followed by a
    swap‑test style measurement to compare input and reconstruction.
    """
    total_qubits = latent_dim + 2 * num_trash + 1  # latent + trash + ancilla
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Feature map for input (first 224 qubits)
    feature_map = _feature_map(total_qubits, reps=reps)
    qc.append(feature_map, qr[:feature_map.num_qubits])

    # Encode latent subspace
    ansatz = RealAmplitudes(latent_dim + num_trash, reps=reps)
    qc.append(ansatz, qr[:latent_dim + num_trash])

    # Swap test with ancilla
    ancilla = total_qubits - 1
    qc.h(ancilla)
    for i in range(num_trash):
        qc.cswap(ancilla, latent_dim + i, latent_dim + num_trash + i)
    qc.h(ancilla)
    qc.measure(ancilla, cr[0])

    return qc

# Sampler‑based QNN
def AutoencoderQNN(latent_dim: int, num_trash: int = 2, reps: int = 3) -> SamplerQNN:
    """Wraps the quantum autoencoder circuit in a SamplerQNN."""
    qc = QuantumAutoencoderCircuit(latent_dim, num_trash, reps)
    sampler = StatevectorSampler()
    # Interpret as probability of ancilla measurement 0
    def interpret(x: np.ndarray) -> np.ndarray:
        return x[0]  # probability of measuring 0
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=interpret,
        output_shape=(1,),
        sampler=sampler,
    )

# Fast estimator for quantum circuits
class FastBaseEstimator:
    """Evaluates expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds shot noise to deterministic expectation values."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
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
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

# Training routine using COBYLA
def train_quantum_autoencoder(
    qnn: SamplerQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-2,
    seed: int | None = None,
) -> List[float]:
    """Optimizes the parameters of the quantum autoencoder to minimize
    the reconstruction error using a COBYLA optimizer."""
    algorithm_globals.random_seed = seed or 42
    opt = COBYLA()
    param_bounds = [(0, 2 * np.pi)] * len(qnn.parameter_names)
    loss_history: List[float] = []

    def objective(params: np.ndarray) -> float:
        # Compute reconstruction error for batch
        batch = data[:100]  # use a subset for speed
        preds = qnn.predict(batch, params=params)
        # The qnn returns probability of ancilla=0; higher means better reconstruction
        # Convert to MSE‑like loss
        loss = np.mean((preds - 1.0) ** 2)
        return loss

    for _ in range(epochs):
        params, fval, _ = opt.optimize(
            n=len(qnn.parameter_names),
            objective_function=objective,
            initial_point=np.random.uniform(0, 2 * np.pi, size=len(qnn.parameter_names)),
            bounds=param_bounds,
        )
        loss_history.append(fval)
    return loss_history

__all__ = [
    "QuantumAutoencoderCircuit",
    "AutoencoderQNN",
    "FastBaseEstimator",
    "FastEstimator",
    "train_quantum_autoencoder",
]
