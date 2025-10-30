"""HybridEstimatorQNN: a quantum‑classical hybrid estimator using Qiskit.

The implementation builds a variational circuit with two parameters – one representing
the classical input and one representing the trainable weight.  The expectation value
of a Pauli‑Y observable is estimated on a qasm simulator and returned as a regression
output.  This mirrors the classical hybrid model defined in the ML module, allowing
direct comparison of quantum and classical performance.

The class is self‑contained and can be used as a drop‑in replacement for the
`HybridEstimatorQNN` defined in the classical module.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridEstimatorQNN:
    """
    Quantum‑inspired hybrid estimator.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the variational circuit.
    shots : int, default 1000
        Number of shots for the qasm simulator.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1000) -> None:
        # Parameters
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")

        # Variational circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.input_param, range(n_qubits))
        self.circuit.rx(self.weight_param, range(n_qubits))
        self.circuit.measure_all()

        # Observable: Pauli‑Y on all qubits (tensor product)
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # Backend and estimator
        self.backend = Aer.get_backend("qasm_simulator")
        self.estimator = StatevectorEstimator(backend=self.backend)

        # Wrap with EstimatorQNN from Qiskit Machine Learning
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def run(self, inputs: Iterable[float]) -> float:
        """
        Evaluate the expectation value for a single input.

        Parameters
        ----------
        inputs : Iterable[float]
            Iterable containing one input value and one weight value.
            The first element is bound to ``input_param`` and the second to ``weight_param``.

        Returns
        -------
        float
            Estimated expectation value of the observable.
        """
        if len(inputs)!= 2:
            raise ValueError("Expected two values: input and weight.")
        param_binds = {self.input_param: inputs[0], self.weight_param: inputs[1]}
        expectation = self.estimator_qnn.run(param_binds)
        # estimator_qnn.run returns a numpy array of shape (1,), extract scalar
        return float(expectation[0])

__all__ = ["HybridEstimatorQNN"]
