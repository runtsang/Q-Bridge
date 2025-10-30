"""Quantum‑centric part of the hybrid estimator.

Only the quantum branch is implemented here; the classical logic is in the
Python module above.  The quantum estimator uses a StatevectorEstimator
backend for exact expectation values, but the API is compatible with
any Qiskit primitive that implements the ``__call__`` protocol.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info.operators import SparsePauliOp
from typing import Iterable, Sequence, List, Union

def _build_quantum_estimator(
    circuit: QuantumCircuit,
    input_params: Sequence[Parameter],
    weight_params: Sequence[Parameter],
    observables: Sequence["qiskit.quantum_info.operators.base_operator.BaseOperator"]
) -> EstimatorQNN:
    """Instantiate a Qiskit EstimatorQNN with a StatevectorEstimator backend."""
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )

__all__ = ["_build_quantum_estimator"]
