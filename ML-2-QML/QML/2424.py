from __future__ import annotations

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

class CombinedQNN:
    """
    Hybrid quantum‑classical model that delivers both an expectation value
    (regression) and a sampling distribution (classification) from a single
    parameterised 2‑qubit circuit.
    """

    def __init__(self) -> None:
        # Parameter vectors
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build the quantum circuit
        qc = QuantumCircuit(2)
        # Input encoding
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        # Entanglement
        qc.cx(0, 1)
        # Parameterised rotations
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        self.circuit = qc

        # Estimator for regression
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[SparsePauliOp.from_list([("Y" * 2, 1)])],
            input_params=[self.inputs[0]],
            weight_params=[self.weights[0]],
            estimator=self.estimator,
        )

        # Sampler for classification
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def predict(
        self,
        input_vals: list[float],
        weight_vals: list[float],
        shots: int = 1024,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate the hybrid QNN.
        Returns:
            - regression expectation value (float)
            - sampling probability distribution (dict of bitstring -> prob)
        """
        # Build parameter binding
        param_dict = {p: v for p, v in zip(self.inputs, input_vals)}
        param_dict.update({p: v for p, v in zip(self.weights, weight_vals)})

        # Regression
        exp_val = self.estimator_qnn.predict(param_dict)

        # Sampling
        samples = self.sampler_qnn.sample(param_dict, shots=shots)
        probs = {}
        for bitstring in samples:
            probs[bitstring] = probs.get(bitstring, 0) + 1
        for k in probs:
            probs[k] /= shots
        return exp_val, probs

__all__ = ["CombinedQNN"]
