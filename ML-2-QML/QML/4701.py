"""Quantum fraud detection model using a parameterized circuit and the SamplerQNN.

The implementation combines the quantum circuit from the FraudDetection seed with
the estimator utilities from FastBaseEstimator, and uses a COBYLA optimizer
for training.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List, Callable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ----------------------------------------------------------------------
# Estimator utilities (from FastBaseEstimator.py)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Tuple[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Iterable[Tuple[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ----------------------------------------------------------------------
# Quantum fraud detection class
# ----------------------------------------------------------------------
class FraudDetectionHybrid:
    """Quantum implementation that embeds input features into a RealAmplitudes ansatz
    and returns a fraud score via a measurement observable.
    """
    def __init__(self, num_qubits: int = 4, reps: int = 2):
        self.num_qubits = num_qubits
        self.reps = reps
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=1,
            sampler=Sampler()
        )
        self.optimizer = COBYLA(maxiter=200, disp=False)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Basic feature encoding: rotate each qubit by the corresponding input angle
        qc.h(range(self.num_qubits))
        qc.compose(RealAmplitudes(self.num_qubits, reps=self.reps), qc.qubits, inplace=True)
        # Measurement placeholder – actual score derived from sampler
        qc.measure(cr[0], cr[0])
        return qc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample in X."""
        X = np.asarray(X, dtype=np.float64)
        # Scale to [0, π] for rotation angles
        X_scaled = np.clip(X, 0, np.pi)
        param_sets = [tuple(x.flatten()) for x in X_scaled]
        outputs = self.qnn.predict(param_sets)
        probs = np.clip(outputs, 0, 1)
        return probs

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 10,
              lr: float = 0.01) -> List[float]:
        """Simple gradient‑free training using COBYLA to minimize cross‑entropy."""
        history: List[float] = []
        for _ in range(epochs):
            def objective(params: np.ndarray) -> float:
                param_sets = [tuple(params) for _ in range(len(X))]
                preds = self.qnn.predict(param_sets).flatten()
                loss = -np.sum(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
                return loss
            init_params = np.random.uniform(0, np.pi, size=len(self.circuit.parameters))
            opt_params = self.optimizer.minimize(objective, init_params)
            # Rebuild circuit with optimized parameters
            self.circuit = self._build_circuit()
            self.qnn = SamplerQNN(
                circuit=self.circuit,
                weight_params=self.circuit.parameters,
                interpret=lambda x: x,
                output_shape=1,
                sampler=Sampler()
            )
            loss = objective(opt_params)
            history.append(loss)
        return history

    def evaluate_with_noise(self,
                            observables: Iterable[BaseOperator],
                            parameter_sets: Iterable[Tuple[float]],
                            *,
                            shots: int | None = None,
                            seed: int | None = None) -> List[List[complex]]:
        """Wrap the FastBaseEstimator to add shot noise to expectation values."""
        estimator = FastBaseEstimator(self.circuit)
        results = estimator.evaluate(observables, parameter_sets)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FraudDetectionHybrid", "FastBaseEstimator"]
