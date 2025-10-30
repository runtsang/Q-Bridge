"""Quantum neural network with a two‑qubit circuit, entanglement, and an observable suited for regression."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from typing import List


def _build_entangled_circuit(
    input_params: List[Parameter], weight_params: List[Parameter], entangle_depth: int = 1
) -> QuantumCircuit:
    """
    Construct a parameterised circuit with optional entanglement depth.

    Parameters
    ----------
    input_params : List[Parameter]
        Parameters that encode the classical input.
    weight_params : List[Parameter]
        Trainable weight parameters.
    entangle_depth : int
        Number of layers of CNOT entanglement between the two qubits.
    """
    qc = QuantumCircuit(2)

    # Encoding layer – simple RY rotations per qubit
    for q, p in enumerate(input_params):
        qc.ry(p, q)

    # Variational layer with two parameters per qubit
    for q, p in enumerate(weight_params):
        qc.rx(p, q)
        qc.rz(p, q)

    # Entanglement – repeat CNOT pairs
    for _ in range(entangle_depth):
        qc.cx(0, 1)
        qc.cx(1, 0)

    return qc


def EstimatorQNN(
    entangle_depth: int = 1,
    observable: SparsePauliOp | None = None,
) -> EstimatorQNN:
    """
    Return a Qiskit EstimatorQNN with a two‑qubit variational circuit.

    Parameters
    ----------
    entangle_depth : int, default=1
        Depth of the entanglement pattern.
    observable : SparsePauliOp, optional
        Observable to be measured. Defaults to Y⊗Y, matching the seed.
    """
    # Define parameter vectors
    input_vec = ParameterVector("x", 2)   # two classical inputs
    weight_vec = ParameterVector("w", 4)  # four trainable weights

    # Build circuit
    qc = _build_entangled_circuit(list(input_vec), list(weight_vec), entangle_depth)

    # Default observable if none supplied
    if observable is None:
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

    # Instantiate the estimator
    estimator = Estimator()
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(input_vec),
        weight_params=list(weight_vec),
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
