"""Quantum estimator built on a two‑qubit variational circuit.

The QNN uses a parameter‑shuffled circuit with entanglement and
supports training with either the StatevectorEstimator or the
PennyLane gradient backend.  The observable is a weighted sum of Z
measurements on both qubits, making the model suitable for regression.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import pennylane as qml
from pennylane import numpy as pnp


def _build_variational_circuit(
    num_qubits: int,
    depth: int,
    params: list[Parameter],
) -> QuantumCircuit:
    """Construct a depth‑controlled entangled circuit."""
    qc = QuantumCircuit(num_qubits)
    # Layer 1: rotation gates
    for i in range(num_qubits):
        qc.ry(params[0][i], i)
        qc.rx(params[1][i], i)
    # Entangling layer
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    # Repeat
    for d in range(1, depth):
        for i in range(num_qubits):
            qc.ry(params[0][d * num_qubits + i], i)
            qc.rx(params[1][d * num_qubits + i], i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    return qc


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN with a 2‑qubit variational circuit."""
    num_qubits = 2
    depth = 2
    # Create parameters: two sets per layer (input, weight)
    input_params = [Parameter(f"x{i}") for i in range(num_qubits)]
    weight_params = [Parameter(f"w{i}") for i in range(num_qubits * depth * 2)]

    # Build circuit
    qc = _build_variational_circuit(num_qubits, depth, [input_params, weight_params])

    # Observable: weighted sum of Z on both qubits
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn


def pennylane_estimator(
    num_qubits: int = 2,
    depth: int = 2,
    seed: int | None = None,
) -> qml.QNode:
    """Return a PennyLane QNode that can be used as a torch‑compatible layer."""
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
        # Encode inputs as Ry rotations
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RZ(weights[idx], wires=i)
                idx += 1
            # Entanglement
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        # Measurement
        return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

    return circuit


__all__ = ["EstimatorQNN", "pennylane_estimator"]
