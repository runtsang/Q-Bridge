"""Quantum classifier with a parameter‑shift gradient and stochastic‑gradient optimizer."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def _parameter_shift_gradient(
    circuit: QuantumCircuit,
    params: np.ndarray,
    observable: SparsePauliOp,
    eps: float = np.pi,
) -> np.ndarray:
    """
    Compute the parameter‑shift gradient for a single observable.

    Parameters
    ----------
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    params : np.ndarray
        Current parameter values.
    observable : SparsePauliOp
        Observable to differentiate with respect to.
    eps : float, optional
        Shift value. Default is π.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (len(params),).
    """
    grad = np.zeros_like(params)
    for idx in range(len(params)):
        shifted_plus = np.copy(params)
        shifted_minus = np.copy(params)
        shifted_plus[idx] += eps / 2
        shifted_minus[idx] -= eps / 2

        plus_circ = circuit.assign_parameters(dict(zip(circuit.parameters, shifted_plus)))
        minus_circ = circuit.assign_parameters(dict(zip(circuit.parameters, shifted_minus)))

        exp_plus = plus_circ.expectation_value(observable)
        exp_minus = minus_circ.expectation_value(observable)

        grad[idx] = (exp_plus - exp_minus) / (2 * np.sin(eps / 2))
    return grad


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    use_entanglement: bool = True,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    use_entanglement : bool, optional
        If True, add CZ gates between neighboring qubits. Default is True.

    Returns
    -------
    QuantumCircuit
        The variational circuit.
    Iterable
        List of encoding parameters.
    Iterable
        List of variational parameters.
    List[SparsePauliOp]
        Observables for each qubit (Z basis).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data‑encoding layer
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        if use_entanglement:
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

    # Observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


def stochastic_gradient_step(
    circuit: QuantumCircuit,
    params: np.ndarray,
    grad: np.ndarray,
    lr: float,
) -> np.ndarray:
    """
    Update parameters using a simple SGD step.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to be updated.
    params : np.ndarray
        Current parameters.
    grad : np.ndarray
        Gradient vector.
    lr : float
        Learning rate.

    Returns
    -------
    np.ndarray
        Updated parameters.
    """
    return params - lr * grad


__all__ = ["build_classifier_circuit", "_parameter_shift_gradient", "stochastic_gradient_step"]
