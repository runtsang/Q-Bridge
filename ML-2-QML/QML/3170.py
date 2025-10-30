"""Hybrid quantum sampler‑estimator network.

The class SamplerQNN constructs two parameterised quantum circuits
using Qiskit.  The sampler circuit outputs a probability distribution
over two outcomes via a StatevectorSampler.  The estimator circuit
measures the expectation value of a single‑qubit Y observable using
StatevectorEstimator.  The two primitives are combined into a single
HybridQNN object that supports both sampling and estimation in a
single forward pass, mirroring the architecture of the classical
counterpart.
"""

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class SamplerQNN:
    """Quantum hybrid sampler‑estimator network."""
    def __init__(self) -> None:
        # Sampler circuit
        inp = ParameterVector("input", 2)
        w_s = ParameterVector("weight_s", 4)
        self.qc_sampler = QuantumCircuit(2)
        self.qc_sampler.ry(inp[0], 0)
        self.qc_sampler.ry(inp[1], 1)
        self.qc_sampler.cx(0, 1)
        self.qc_sampler.ry(w_s[0], 0)
        self.qc_sampler.ry(w_s[1], 1)
        self.qc_sampler.cx(0, 1)
        self.qc_sampler.ry(w_s[2], 0)
        self.qc_sampler.ry(w_s[3], 1)

        # Estimator circuit
        inp_e = Parameter("input_e")
        w_e = Parameter("weight_e")
        self.qc_estimator = QuantumCircuit(1)
        self.qc_estimator.h(0)
        self.qc_estimator.ry(inp_e, 0)
        self.qc_estimator.rx(w_e, 0)

        # Observable for estimator
        observable = SparsePauliOp.from_list([("Y", 1)])

        # Primitives
        sampler = StatevectorSampler()
        estimator = StatevectorEstimator()

        # Qiskit Machine Learning wrappers
        self.sampler_qnn = QSamplerQNN(
            circuit=self.qc_sampler,
            input_params=inp,
            weight_params=w_s,
            sampler=sampler,
        )
        self.estimator_qnn = QEstimatorQNN(
            circuit=self.qc_estimator,
            observables=observable,
            input_params=[inp_e],
            weight_params=[w_e],
            estimator=estimator,
        )

    def sample(self, input_values: list[float], weight_values: list[float]) -> list[float]:
        """Return probability vector from the sampler circuit."""
        return self.sampler_qnn(input_values, weight_values)

    def estimate(self, input_value: float, weight_value: float) -> float:
        """Return expectation value from the estimator circuit."""
        return self.estimator_qnn([input_value], [weight_value])

__all__ = ["SamplerQNN"]
