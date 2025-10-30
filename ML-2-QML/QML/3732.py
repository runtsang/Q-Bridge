"""Combined quantum neural network that merges EstimatorQNN and FCL quantum designs.

The circuit operates on two qubits: qubit 0 encodes the input feature via a
parameterized Ry gate; qubit 1 holds the weight parameter and is entangled
with qubit 0.  After a sequence of rotations and an entangling CX gate the
observable Y⊗I is measured.  The resulting expectation value serves as the
output of the quantum neural network.  The construction can be used with
qiskit‑machine‑learning's EstimatorQNN class for hybrid training.

Usage
-----
>>> from EstimatorQNN__gen024 import EstimatorQNNQuantum
>>> qnn = EstimatorQNNQuantum()
>>> y = qnn.predict([[0.5], [1.2]])
"""

from __future__ import annotations

from typing import List
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNQuantum:
    """
    Quantum neural network that fuses the EstimatorQNN single‑qubit circuit
    with the FCL parameterized circuit.  It uses two qubits, a single
    input parameter, and a single weight parameter.
    """
    def __init__(self, shots: int = 1000):
        self.shots = shots
        self.estimator = StatevectorEstimator()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        # Parameters
        input_param = Parameter("input")
        weight_param = Parameter("weight")

        # Build circuit
        qc = QuantumCircuit(2)
        # Encode input on qubit 0
        qc.h(0)
        qc.ry(input_param, 0)
        # Encode weight on qubit 1
        qc.h(1)
        qc.ry(weight_param, 1)
        # Entangle
        qc.cx(0, 1)
        # Rotate qubit 0 with weight again (mimicking EstimatorQNN rx)
        qc.rx(weight_param, 0)

        # Observable Y on qubit 0
        observable = SparsePauliOp.from_list([("Y" + "I", 1)])

        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=self.estimator,
        )

    def predict(self, inputs: List[List[float]]) -> List[float]:
        """
        Evaluate the quantum circuit for a list of input values.
        Each input is a list of a single float (the input parameter).
        The weight is left as a trainable parameter; for prediction we
        use the current value of the weight parameter in the estimator.
        """
        preds = []
        for inp in inputs:
            # Bind input and keep current weight
            bound = {self.qnn.input_params[0]: inp[0]}
            result = self.estimator.run([self.qnn.circuit], [bound])
            preds.append(result[0])
        return preds

    def __call__(self, inputs: List[List[float]]) -> List[float]:
        return self.predict(inputs)

__all__ = ["EstimatorQNNQuantum"]
