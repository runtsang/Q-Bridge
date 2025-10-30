"""Quantum hybrid sampler & estimator network.

The circuit shares parameters for both a two‑qubit sampler and a single‑qubit
estimator.  It demonstrates how a single variational ansatz can be used
simultaneously for generative sampling and expectation‑value regression,
leveraging Qiskit Machine Learning primitives.
"""

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

class SamplerEstimatorQNN:
    """Hybrid QNN returning both probability distribution and expectation."""

    def __init__(self) -> None:
        # 2 inputs for sampler, 1 for estimator
        self.input_params = ParameterVector("input", 3)
        # 6 weight parameters: 4 for sampler, 2 for estimator
        self.weight_params = ParameterVector("weight", 6)

        qc = QuantumCircuit(3)

        # Sampling part (qubits 0 & 1)
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)

        # Estimation part (qubit 2)
        qc.ry(self.input_params[2], 2)

        # Weight rotations
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.ry(self.weight_params[2], 2)
        qc.cx(0, 1)
        qc.ry(self.weight_params[3], 0)
        qc.ry(self.weight_params[4], 1)
        qc.ry(self.weight_params[5], 2)

        self.qc = qc

        # SamplerQNN
        sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc,
            input_params=self.input_params[:2],
            weight_params=self.weight_params[:4],
            sampler=sampler,
        )

        # EstimatorQNN
        estimator = StatevectorEstimator()
        observable = SparsePauliOp.from_list([("Y", 1)])  # Pauli Y on qubit 2
        self.estimator_qnn = QEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.input_params[2]],
            weight_params=self.weight_params[4:],
            estimator=estimator,
        )

    def __call__(self, x):
        """Return dictionary with'sample' and 'estimate' given a batch of 3 inputs."""
        return {
            "sample": self.sampler_qnn(x[:, :2]),
            "estimate": self.estimator_qnn(x[:, 2:]),
        }

__all__ = ["SamplerEstimatorQNN"]
