"""Variational EstimatorQNN with multi‑qubit ansatz and configurable depth.

The quantum network mirrors the classical EstimatorQNN signature but replaces the
feed‑forward layers with a parameterised quantum circuit.  The ansatz is a simple
repeat of single‑qubit rotations followed by a chain of CNOTs, providing a
scalable expressivity while remaining easy to compile on simulators.
"""

from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import AerEstimator


def _build_ansatz(num_qubits: int, depth: int) -> tuple[QuantumCircuit, List[Parameter]]:
    """Create a depth‑controlled variational circuit."""
    qc = QuantumCircuit(num_qubits)
    params: List[Parameter] = []

    for d in range(depth):
        for q in range(num_qubits):
            p_ry = Parameter(f"ry_{d}_{q}")
            p_rz = Parameter(f"rz_{d}_{q}")
            qc.ry(p_ry, q)
            qc.rz(p_rz, q)
            params.extend([p_ry, p_rz])
        # Entangling layer: linear chain of CNOTs
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    return qc, params


def _observable(num_qubits: int, observable_type: str) -> SparsePauliOp:
    """Return a Pauli observable to be measured."""
    if observable_type.lower() == "y":
        pauli_string = "Y" * num_qubits
    elif observable_type.lower() == "z":
        pauli_string = "Z" * num_qubits
    else:
        raise ValueError(f"Unsupported observable type: {observable_type}")

    return SparsePauliOp.from_list([(pauli_string, 1.0)])


def EstimatorQNN(
    num_qubits: int = 3,
    depth: int = 2,
    observable: str = "y",
    estimator=None,
) -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN instance.

    Parameters
    ----------
    num_qubits : int, default 3
        Number of qubits in the variational circuit.
    depth : int, default 2
        Depth of the ansatz (number of rotation + CNOT layers).
    observable : str, default "y"
        Pauli string to observe; currently 'y' or 'z'.
    estimator : Primitive | None, default None
        Quantum primitive used for state‑vector evaluation.  If ``None``, a
        AerEstimator is instantiated.

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
        The wrapped variational quantum neural network ready for training.
    """
    qc, params = _build_ansatz(num_qubits, depth)
    obs = _observable(num_qubits, observable)

    if estimator is None:
        estimator = AerEstimator()

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=[],  # No classical input parameters – variational only
        weight_params=params,
        estimator=estimator,
    )


__all__ = ["EstimatorQNN"]
