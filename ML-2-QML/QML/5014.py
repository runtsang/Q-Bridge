import numpy as np
from typing import Sequence, List, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

class HybridSamplerEstimatorQNN:
    """
    Quantum counterpart of HybridSamplerEstimatorQNN.  It builds a
    parameterised sampler circuit (2 qubits) and a 1‑qubit estimator
    circuit, both driven by a shared input vector.  The `evaluate`
    method returns the sampler probabilities and the estimator
    expectation, optionally adding Gaussian shot noise.
    """

    def __init__(self, shots: int | None = None) -> None:
        # Parameter vectors
        self.inputs = ParameterVector("input", 2)
        self.sampler_weights = ParameterVector("weight_s", 4)
        self.estimator_params = ParameterVector("estimator_par", 2)

        # Sampler circuit
        qc_sampler = QuantumCircuit(2)
        qc_sampler.ry(self.inputs[0], 0)
        qc_sampler.ry(self.inputs[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[0], 0)
        qc_sampler.ry(self.sampler_weights[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[2], 0)
        qc_sampler.ry(self.sampler_weights[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc_sampler,
            input_params=self.inputs,
            weight_params=self.sampler_weights,
            sampler=sampler,
        )

        # Estimator circuit
        qc_est = QuantumCircuit(1)
        qc_est.h(0)
        qc_est.ry(self.estimator_params[0], 0)
        qc_est.rx(self.estimator_params[1], 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc_est,
            observables=observable,
            input_params=[self.estimator_params[0]],
            weight_params=[self.estimator_params[1]],
            estimator=estimator,
        )

        self.shots = shots

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Sequence[Callable[[dict], float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the quantum network for a list of parameter sets.

        The expected format for each parameter set is:
            [input0, input1, sampler_w0, sampler_w1, sampler_w2, sampler_w3,
             est_inp, est_wt]
        """
        if observables is None:
            observables = [
                lambda out: np.mean(out["sampler"]),
                lambda out: out["estimator"],
            ]

        results: List[List[float]] = []

        for params in parameter_sets:
            if len(params)!= 8:
                raise ValueError("Each parameter set must contain 8 floats.")
            # Bind parameters
            mapping = {
                self.inputs[0]: params[0],
                self.inputs[1]: params[1],
                self.sampler_weights[0]: params[2],
                self.sampler_weights[1]: params[3],
                self.sampler_weights[2]: params[4],
                self.sampler_weights[3]: params[5],
                self.estimator_params[0]: params[6],
                self.estimator_params[1]: params[7],
            }
            bound_sampler = self.sampler_qnn.circuit.assign_parameters(mapping, inplace=False)
            bound_estimator = self.estimator_qnn.circuit.assign_parameters(mapping, inplace=False)

            # Sampler probabilities
            state = self.sampler_qnn.sampler.sample(bound_sampler)
            probs = np.abs(state)**2
            # Estimator expectation
            est_val = self.estimator_qnn.estimator.sample(bound_estimator)[0]

            out = {"sampler": probs, "estimator": est_val}
            row = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, np.ndarray):
                    val = float(val)
                row.append(float(val))
            results.append(row)

        # Add Gaussian shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
                noisy.append(noisy_row)
            results = noisy

        return results

__all__ = ["HybridSamplerEstimatorQNN"]
