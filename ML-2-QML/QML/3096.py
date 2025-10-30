"""Hybrid quantum neural network combining sampler and estimator circuits."""

from __future__ import annotations

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator


class HybridQNN:
    """
    Quantum neural network that exposes both a sampler and an estimator.
    The sampler outputs a two‑qubit probability distribution; the estimator
    evaluates the expectation of a Pauli‑Y observable on a single qubit.
    Both components are parametrised and can be trained jointly.
    """

    def __init__(self) -> None:
        # --- Sampler circuit (2 qubits) ------------------------------------
        sampler_inputs = ParameterVector("s_in", 2)
        sampler_weights = ParameterVector("s_w", 4)

        qc_sampler = QuantumCircuit(2)
        qc_sampler.ry(sampler_inputs[0], 0)
        qc_sampler.ry(sampler_inputs[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(sampler_weights[0], 0)
        qc_sampler.ry(sampler_weights[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(sampler_weights[2], 0)
        qc_sampler.ry(sampler_weights[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc_sampler,
            input_params=sampler_inputs,
            weight_params=sampler_weights,
            sampler=sampler,
        )

        # --- Estimator circuit (1 qubit) -----------------------------------
        estimator_input = Parameter("e_in")
        estimator_weight = Parameter("e_w")

        qc_estimator = QuantumCircuit(1)
        qc_estimator.h(0)
        qc_estimator.ry(estimator_input, 0)
        qc_estimator.rx(estimator_weight, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])

        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc_estimator,
            observables=observable,
            input_params=[estimator_input],
            weight_params=[estimator_weight],
            estimator=estimator,
        )

    def sample(self, inputs: list[float]) -> list[float]:
        """Return probabilities from the sampler QNN."""
        return self.sampler_qnn.evaluate(inputs)

    def estimate(self, inputs: list[float]) -> float:
        """Return expectation value from the estimator QNN."""
        return self.estimator_qnn.evaluate(inputs)[0]


__all__ = ["HybridQNN"]
