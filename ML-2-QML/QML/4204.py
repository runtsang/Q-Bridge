"""Quantum hybrid classifier that mirrors the classical API.

The class builds a variational circuit with data encoding, a configurable
depth of parameterized rotations and CZ entanglers, and a measurement
operator that produces a vector of expectation values.  The static method
`build_classifier_circuit` returns the circuit and its associated
parameters, enabling direct comparison with the classical implementation.

The module also contains a quantum sampler (SamplerQNN) and regression
utilities analogous to the classical side.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit_machine_learning.neural_networks as qml_nn
from qiskit.primitives import StatevectorSampler as Sampler

__all__ = [
    "HybridClassifierModel",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
]


class HybridClassifierModel:
    """Quantum hybrid classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (input dimensionality).
    depth : int, default=2
        Number of variational layers after the encoding block.
    """

    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = (
            self.build_classifier_circuit(num_qubits, depth)
        )

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a data‑encoding ansatz followed by a depth‑controlled variational block.

        Returns
        -------
        circuit : QuantumCircuit
            The variational circuit.
        encoding : list of ParameterVector
            Parameters used for data encoding.
        weights : list of ParameterVector
            Trainable variational parameters.
        observables : list of SparsePauliOp
            Pauli‑Z measurements on each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        # Data encoding: RX rotations
        for qubit in range(num_qubits):
            qc.rx(encoding[qubit], qubit)

        # Variational block
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurement operators
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return qc, [encoding], [weights], observables

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_parameters(self) -> Tuple[List[ParameterVector], List[ParameterVector]]:
        """Return encoding and weight parameters."""
        return self.encoding, self.weights


def SamplerQNN() -> qml_nn.SamplerQNN:
    """Quantum sampler that mimics the classical sampler.

    The sampler uses a small 2‑qubit circuit with parameterized rotations
    and a CNOT gate.  It is wrapped in Qiskit Machine Learning's
    SamplerQNN class for easy integration into variational workflows.
    """
    # Define parameters
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    # Build circuit
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    # Sampler primitive
    sampler = Sampler()
    sampler_qnn = qml_nn.SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn


def generate_superposition_data(
    num_wires: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a superposition state dataset.

    The states are of the form
        cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩
    and the labels are a noisy sinusoid of θ and ϕ.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum regression experiments.

    Returns a dictionary with the quantum state and the target scalar.
    """

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }
