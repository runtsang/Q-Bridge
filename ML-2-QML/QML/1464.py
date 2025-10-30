"""Quantum incremental classifier with variational ansatz and expectation‑to‑logits mapping."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import torch

def build_classifier_circuit(num_qubits: int, depth: int, num_classes: int = 3
                             ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz and return its metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (also number of input features).
    depth : int
        Depth of the ansatz.
    num_classes : int, default=3
        Number of classification labels.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised circuit ready for execution.
    encoding : Iterable
        ParameterVector for data encoding.
    weights : Iterable
        ParameterVector for variational angles.
    observables : List[SparsePauliOp]
        Pauli‑Z observables used to extract expectation values.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for i, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[i], qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def compute_expectations(circuit: QuantumCircuit,
                         param_values: np.ndarray,
                         observables: List[SparsePauliOp],
                         backend=Aer.get_backend("aer_simulator"),
                         shots: int = 1024) -> np.ndarray:
    """
    Evaluate the circuit with the supplied parameters and return the
    expectation values for each observable.

    Parameters
    ----------
    circuit : QuantumCircuit
        The variational circuit.
    param_values : np.ndarray
        Array of shape (len(encoding)+len(weights),) containing the numerical values
        for all parameters.
    observables : List[SparsePauliOp]
        List of observables to measure.
    backend : Aer backend, optional
        The simulator to use.
    shots : int, default=1024
        Number of measurement shots.

    Returns
    -------
    expectations : np.ndarray
        Expectation values, shape (len(observables),).
    """
    param_dict = {p: v for p, v in zip(circuit.parameters, param_values)}
    bound_qc = circuit.bind_parameters(param_dict)
    job = execute(bound_qc, backend, shots=shots, memory=False)
    result = job.result()
    expectations = []
    for op in observables:
        expectations.append(result.get_expectation_value(op, bound_qc))
    return np.array(expectations)

class QuantumClassifier:
    """
    Lightweight wrapper that maps circuit expectations to class logits.

    The classical linear head is implemented with PyTorch tensors so that
    the whole model can be differentiated with respect to the quantum
    parameters using the parameter‑shift rule.
    """
    def __init__(self, num_qubits: int, depth: int, num_classes: int = 3,
                 backend=None):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth, num_classes)
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.num_params = len(self.circuit.parameters)
        self.linear_W = torch.nn.Parameter(torch.randn(num_classes, len(self.observables)))
        self.linear_b = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, data: np.ndarray) -> torch.Tensor:
        """
        Compute logits for a batch of feature vectors.

        Parameters
        ----------
        data : np.ndarray, shape (batch, num_qubits)

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        batch_size = data.shape[0]
        logits = []
        for i in range(batch_size):
            param_values = np.concatenate([data[i], np.zeros(self.num_params - data.shape[1])])
            expectations = compute_expectations(self.circuit, param_values,
                                                 self.observables,
                                                 backend=self.backend)
            exp_tensor = torch.tensor(expectations, dtype=torch.float32)
            logit = exp_tensor @ self.linear_W.t() + self.linear_b
            logits.append(logit)
        return torch.stack(logits)

    def parameters(self):
        """Yield all learnable parameters."""
        return [self.linear_W, self.linear_b] + list(self.circuit.parameters)

__all__ = ["build_classifier_circuit", "compute_expectations", "QuantumClassifier"]
