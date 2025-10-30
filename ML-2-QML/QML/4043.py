"""Hybrid quantum estimator that mirrors the classical API.

Features
--------
- A fully‑connected style parameterised circuit (one qubit) that
  accepts an input parameter and a weight parameter.
- The circuit is wrapped in a qiskit Machine‑Learning EstimatorQNN
  object.
- A FastBaseEstimator class evaluates the circuit for many parameter
  sets using the Statevector simulator, identical to the classical
  counterpart.
- Optional shot‑noise can be added via FastEstimator.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List

# --------------------------------------------------------------------------- #
# Quantum FCL surrogate
# --------------------------------------------------------------------------- #
class QuantumFCL:
    """A simple 1‑qubit parameterised circuit mimicking the classical FCL.

    Parameters
    ----------
    backend : Backend
        The qiskit backend to execute the circuit on.
    shots : int
        Number of shots for the state‑vector estimator.
    """
    def __init__(self, backend, shots: int = 100):
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")

        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the circuit on the backend and return the expectation of Y."""
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

# --------------------------------------------------------------------------- #
# Core quantum estimator
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> QEstimatorQNN:
    """Return a qiskit Machine‑Learning EstimatorQNN instance.

    The circuit consists of a single qubit with a H, an input rotation
    (parameterised by ``input1``) and a trainable rotation (``weight1``).
    The observable is Y on the single qubit.  The function returns the
    EstimatorQNN wrapper so that it can be used like the classical
    EstimatorQNN.
    """
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    return QEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )

# --------------------------------------------------------------------------- #
# Fast evaluation utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised circuit.

    The API matches the classical FastBaseEstimator so that the same
    training loop can be applied to both backends.
    """
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
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            results.append([state.expectation_value(obs) for obs in observables])
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimations."""
    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
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
            noisy.append([complex(rng.normal(float(v.real), max(1e-6, 1 / shots))) for v in row])
        return noisy

__all__ = ["EstimatorQNN", "FastBaseEstimator", "FastEstimator", "QuantumFCL"]
