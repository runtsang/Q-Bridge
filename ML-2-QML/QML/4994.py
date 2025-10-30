"""
Quantum‑circuit estimator that wraps Qiskit’s EstimatorQNN.

It shares the public API with the classical EstimatorQNN, enabling
side‑by‑side benchmarking.  The class builds a parameterised
circuit, couples it to a `StatevectorEstimator`, and optionally
adds Gaussian shot noise via the FastEstimator primitive.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Local utilities
from.FastEstimator import FastEstimator


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Same signature as the reference, but re‑implemented locally."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


class EstimatorQNN:
    """
    Quantum estimator that mimics the public API of the classical EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the parameterised circuit.
    depth : int, default=2
        Depth of the ansatz.
    shots : int, default=None
        If provided, Gaussian shot noise is added to the output.
    seed : int, default=None
        Random seed for shot noise.
    """

    def __init__(self, num_qubits: int, depth: int = 2,
                 shots: int | None = None, seed: int | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.seed = seed

        # Build the underlying Qiskit circuit and EstimatorQNN
        circuit, enc_params, var_params, observables = build_classifier_circuit(
            num_qubits, depth
        )
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=enc_params,
            weight_params=var_params,
            estimator=estimator,
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        Parameters are expected in the order [input_params, weight_params].
        """
        # Delegate to the wrapped EstimatorQNN
        raw = self.estimator_qnn.evaluate(observables, parameter_sets)
        # Convert list of lists of complex to Python float list
        results = [[float(v) for v in row] for row in raw]
        return results

    # ------------------------------------------------------------------
    # Batch evaluation with optional noise
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Iterable[Iterable[float]] | None = None,
    ) -> List[List[float]]:
        """
        Convenience wrapper that adds Gaussian shot noise if `shots` is set.
        """
        estimator = FastEstimator(self)  # reuse FastEstimator logic
        raw = estimator.evaluate(
            observables=observables,
            parameter_sets=parameter_sets,
        )
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Utility to expose the underlying Qiskit circuit
    # ------------------------------------------------------------------
    @property
    def circuit(self) -> QuantumCircuit:
        return self.estimator_qnn.circuit

    @property
    def observables(self) -> List[SparsePauliOp]:
        return list(self.estimator_qnn.observables)
