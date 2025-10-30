import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

class HybridEstimatorQNN:
    """
    Quantum implementation of the hybrid estimator.
    It builds a composite variational circuit that consists of a sampler,
    a self‑attention block, and a regression estimator.
    """

    def __init__(self):
        self.backend = qiskit.Aer.get_backend("statevector_simulator")
        # Build individual sub‑circuits
        self.sampler_circuit = self._build_sampler()
        self.attention_circuit = self._build_attention()
        self.estimator = self._build_estimator()

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
        return qc

    def _build_attention(self):
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        rot = ParameterVector("rot", 12)
        ent = ParameterVector("ent", 3)
        for i in range(4):
            circuit.rx(rot[3*i], i)
            circuit.ry(rot[3*i+1], i)
            circuit.rz(rot[3*i+2], i)
        for i in range(3):
            circuit.crx(ent[i], i, i+1)
        circuit.measure(qr, cr)
        return circuit

    def _build_estimator(self):
        inp = Parameter("input")
        wgt = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(inp, 0)
        qc.rx(wgt, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = Estimator()
        return QEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[inp],
            weight_params=[wgt],
            estimator=estimator
        )

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the composite quantum model: sampler → attention → estimator.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, 2) with classical features.

        Returns
        -------
        np.ndarray
            Regression outputs produced by the quantum estimator.
        """
        # Placeholder: return random outputs to illustrate interface
        return np.random.rand(inputs.shape[0], 1)

__all__ = ["HybridEstimatorQNN"]
