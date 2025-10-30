"""Hybrid quantum sampler‑estimator built on a 2‑qubit 192‑parameter circuit.

The module exposes a reusable SamplerQNN and EstimatorQNN pair, as well as a
wrapper that runs both operations in a single call.  The circuit is
parameterised with two input parameters and 192 weight parameters, grouped
into repeated entangling layers to capture richer quantum features.
"""

from __future__ import annotations

from typing import Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


def _build_hybrid_circuit() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Construct a 2‑qubit circuit with 192 weight parameters and 2 input parameters.
    The design follows the seeds but expands depth to 48 entangling blocks.
    """
    inputs = ParameterVector("x", 2)
    weights = ParameterVector("w", 192)

    qc = QuantumCircuit(2)

    # Input encoding
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)

    # 48 repeated entangling layers (4 Ry gates + a CX pair)
    for i in range(0, 192, 4):
        qc.cx(0, 1)
        qc.ry(weights[i], 0)
        qc.ry(weights[i + 1], 1)
        qc.cx(0, 1)
        qc.ry(weights[i + 2], 0)
        qc.ry(weights[i + 3], 1)

    return qc, inputs, weights


def SamplerQNN() -> QiskitSamplerQNN:
    """
    Return a Qiskit SamplerQNN instance configured with the 192‑parameter circuit.
    """
    qc, inputs, weights = _build_hybrid_circuit()
    sampler = StatevectorSampler()
    return QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )


def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Return a Qiskit EstimatorQNN instance that measures the Y observable
    on the first qubit of the same circuit.
    """
    qc, inputs, weights = _build_hybrid_circuit()
    observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[inputs[0]],
        weight_params=[weights[0]],
        estimator=estimator,
    )


class HybridQuantumSamplerEstimator:
    """
    Wrapper that runs both sampling and expectation evaluation in a single
    forward pass.  Useful for joint training or hybrid inference.
    """

    def __init__(self) -> None:
        self.sampler_qnn = SamplerQNN()
        self.estimator_qnn = EstimatorQNN()
        self.circuit, *_ = _build_hybrid_circuit()

    def sample_and_estimate(
        self, input_vals: List[float], weight_vals: List[float]
    ) -> tuple[List[float], float]:
        """
        Execute the hybrid circuit with the provided parameters.

        Args:
            input_vals: List of two floats for the input parameters.
            weight_vals: List of 192 floats for the weight parameters.

        Returns:
            probs: Sampling probabilities for the two‑qubit computational basis.
            expectation: Expectation value of the Y observable on the first qubit.
        """
        probs = self.sampler_qnn.sample(input_vals, weight_vals)
        expectation = self.estimator_qnn.evaluate([input_vals[0]], [weight_vals[0]])
        return probs, expectation


__all__ = [
    "HybridQuantumSamplerEstimator",
    "SamplerQNN",
    "EstimatorQNN",
]
