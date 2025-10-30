from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

class HybridSamplerEstimatorQNN:
    """
    Hybrid quantum neural network that includes a sampler circuit (2 qubits) and an
    estimator circuit (1 qubit).  The sampler generates a probability distribution
    over two outcomes, which is then used as the input to the estimator to predict
    a real‑valued quantity.
    """
    def __init__(self):
        # Sampler circuit
        self.sampler_inputs = ParameterVector("input_s", 2)
        self.sampler_weights = ParameterVector("weight_s", 4)
        qc_s = QuantumCircuit(2)
        qc_s.ry(self.sampler_inputs[0], 0)
        qc_s.ry(self.sampler_inputs[1], 1)
        qc_s.cx(0, 1)
        for i, w in enumerate(self.sampler_weights):
            qc_s.ry(w, i % 2)
        qc_s.cx(0, 1)

        sampler = Sampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc_s,
            input_params=self.sampler_inputs,
            weight_params=self.sampler_weights,
            sampler=sampler,
        )

        # Estimator circuit
        self.estimator_inputs = Parameter("input_e")
        self.estimator_weights = Parameter("weight_e")
        qc_e = QuantumCircuit(1)
        qc_e.h(0)
        qc_e.ry(self.estimator_inputs, 0)
        qc_e.rx(self.estimator_weights, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])

        estimator = Estimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=qc_e,
            observables=observable,
            input_params=[self.estimator_inputs],
            weight_params=[self.estimator_weights],
            estimator=estimator,
        )

    def run_sampler(self, inputs: list[float], weights: list[float]) -> list[float]:
        """
        Execute the sampler circuit with given parameters and return a probability
        vector.
        """
        param_dict = {p: v for p, v in zip(self.sampler_inputs, inputs)}
        param_dict.update({p: v for p, v in zip(self.sampler_weights, weights)})
        return self.sampler_qnn.run(param_dict)

    def run_estimator(self, input_s: list[float], weight_e: float) -> float:
        """
        Execute the estimator circuit given the sampler output (as an input)
        and a weight parameter to produce a real‑valued expectation.
        """
        param_dict = {self.estimator_inputs: input_s[0], self.estimator_weights: weight_e}
        return self.estimator_qnn.run(param_dict)[0]

    def run(self, sampler_inputs: list[float], sampler_weights: list[float], weight_e: float) -> float:
        """
        Full hybrid workflow: sample then estimate.
        """
        probs = self.run_sampler(sampler_inputs, sampler_weights)
        # Use the first probability as the estimator input
        return self.run_estimator(probs, weight_e)
