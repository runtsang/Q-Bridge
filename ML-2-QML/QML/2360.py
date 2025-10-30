"""Quantum estimator that consumes the classical metadata and performs
parameter‑shift evaluation.

It builds a variational circuit with encoding and weight parameters,
uses a StatevectorEstimator to compute expectation values of the
observables, and exposes a ``predict`` method that accepts a numpy
array of feature vectors.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def build_quantum_circuit(num_qubits: int,
                          depth: int) -> Tuple[QuantumCircuit,
                                                Iterable[Parameter],
                                                Iterable[Parameter],
                                                List[SparsePauliOp]]:
    """
    Build a layered ansatz that mirrors the classical builder.

    Parameters
    ----------
    num_qubits: int
        Number of qubits/features.
    depth: int
        Number of variational layers.

    Returns
    -------
    circuit: QuantumCircuit
        The variational circuit with encoding and weight parameters.
    encoding: Iterable[Parameter]
        Parameters used to encode the classical inputs.
    weights: Iterable[Parameter]
        Variational parameters.
    observables: List[SparsePauliOp]
        Observables whose expectation values form the output.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, encoding, weights, observables

class EstimatorQNNHybridQML:
    """
    Quantum estimator that wraps Qiskit’s EstimatorQNN.

    The class accepts a classical ``EstimatorQNNHybrid`` instance to
    reuse its metadata.  The prediction routine maps a batch of
    feature vectors into the quantum circuit, evaluates the
    expectation values and returns the result as a NumPy array.
    """

    def __init__(self,
                 classical: "EstimatorQNNHybrid",
                 depth: int = 2) -> None:
        """
        Parameters
        ----------
        classical: EstimatorQNNHybrid
            The classical counterpart that supplies encoding indices,
            weight sizes and observables.
        depth: int, default 2
            Depth of the variational ansatz.
        """
        num_qubits = len(classical.encoding)
        circuit, encoding, weights, observables = build_quantum_circuit(
            num_qubits, depth
        )
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=list(encoding),
            weight_params=list(weights),
            estimator=estimator,
        )
        # store the classical weight parameters to initialise the
        # quantum circuit (optional, but useful for hybrid training)
        self.classical = classical

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit for each sample in X.

        Parameters
        ----------
        X: np.ndarray of shape (batch, num_features)
            Input feature matrix.

        Returns
        -------
        y_pred: np.ndarray of shape (batch, output_dim)
            The expectation values of the observables.
        """
        batch_size = X.shape[0]
        # prepare input parameters dict
        input_dicts = [{f"x[{i}]": val for i, val in enumerate(row)}
                       for row in X]
        # weight params are left symbolic; here we initialise them to zero
        weight_dict = {f"theta[{i}]": 0.0 for i in range(len(self.estimator_qnn.weight_params))}

        # evaluate each sample
        preds = []
        for inp in input_dicts:
            params = {**inp, **weight_dict}
            res = self.estimator_qnn(params)
            preds.append(res)
        return np.array(preds)

__all__ = ["EstimatorQNNHybridQML", "build_quantum_circuit"]
