from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class HybridQuantumModel:
    """Hybrid quantum circuit builder for classification or regression.

    The class exposes a static method ``build_classifier_circuit`` that returns a
    Qiskit ``QuantumCircuit`` together with encoding parameters, variational
    parameters and observables.  The method accepts a ``task`` argument to
    switch between classification (multi‑output Z observables) and regression
    (single Z observable).  The circuit implements a simple data‑uploading
    ansatz with Ry rotations and CZ entangling gates.

    The module also contains helper functions to generate the same superposition
    data used by the classical regression seed.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        task: str = "classification",
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit encoding and variational parameters.

        Parameters
        ----------
        num_qubits : int
            Number of qubits / features.
        depth : int
            Depth of the ansatz.
        task : str, optional
            Either ``"classification"`` or ``"regression"``. For regression we return
            a single Z observable.

        Returns
        -------
        circuit : QuantumCircuit
            The variational circuit.
        encoding : List[ParameterVector]
            Encoding parameters.
        weights : List[ParameterVector]
            Variational parameters.
        observables : List[SparsePauliOp]
            Observables to measure.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        if task == "classification":
            observables = [
                SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                for i in range(num_qubits)
            ]
        else:  # regression
            observables = [SparsePauliOp("Z" + "I" * (num_qubits - 1))]

        return circuit, list(encoding), list(weights), observables

def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Regression targets of shape (samples,).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

__all__ = ["HybridQuantumModel", "generate_superposition_data"]
