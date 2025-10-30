"""Quantum counterpart of HybridEstimatorQNN using Qiskit and the EstimatorQNN wrapper.

The circuit encodes input features via Ry rotations, applies a RandomLayer
to simulate a quantum feature map, and measures all qubits in the Z basis.
The estimator returns expectation values that are fed into a classical
linear readout, mirroring the hybrid's quantum‑inspired block."""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RandomLayer
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator
from typing import List


def HybridEstimatorQNN(num_qubits: int) -> EstimatorQNN:
    """Return a Qiskit EstimatorQNN that parallels the classical HybridEstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, matching the input feature dimension of the classical model.

    Returns
    -------
    EstimatorQNN
        A variational quantum neural network ready for training with Qiskit.
    """
    # --- Input encoding: one Ry per qubit ---
    input_params: List[Parameter] = [Parameter(f"input_{i}") for i in range(num_qubits)]

    # --- Variational block: RandomLayer simulates a quantum feature map ---
    rl = RandomLayer(num_qubits, reps=3)
    weight_params = rl.parameters

    # Build the circuit
    qc = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(input_params):
        qc.ry(param, qubit)
    qc.append(rl, range(num_qubits))

    # Measure all qubits in the Z basis
    qc.measure_all()

    # Observable: tensor product of Pauli Z on all qubits
    obs = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    estimator = StateEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


__all__ = ["HybridEstimatorQNN"]
