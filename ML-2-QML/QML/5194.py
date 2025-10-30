from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

class HybridEstimatorQNNQML:
    """
    Quantum‑centric wrapper that exposes the same four modes as the classical
    counterpart.  Each mode builds a Qiskit circuit, attaches an estimator or
    sampler primitive, and provides a simple ``evaluate`` interface.
    """
    def __init__(self, mode: str = "estimator"):
        self.mode = mode.lower()
        if self.mode == "estimator":
            self.estimator = self._build_estimator()
        elif self.mode == "sampler":
            self.sampler = self._build_sampler()
        else:
            raise ValueError(f"Unsupported QML mode: {mode}")

    # --------------------------------------------------------------------- #
    # Estimator construction – adapted from EstimatorQNN.py
    # --------------------------------------------------------------------- #
    def _build_estimator(self):
        # One input, one trainable weight
        input_param = Parameter("input")
        weight_param = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return QEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )

    # --------------------------------------------------------------------- #
    # Sampler construction – adapted from SamplerQNN.py
    # --------------------------------------------------------------------- #
    def _build_sampler(self):
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        return QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    # --------------------------------------------------------------------- #
    # Public evaluation interface
    # --------------------------------------------------------------------- #
    def evaluate(self, input_data: dict):
        """
        Dispatches to the underlying Qiskit primitive.
        ``input_data`` should be a dict mapping parameter names to values.
        """
        if self.mode == "estimator":
            return self.estimator.evaluate(input_data)
        elif self.mode == "sampler":
            return self.sampler.sample(input_data)
        else:
            raise RuntimeError("Evaluation is only defined for supported modes.")
