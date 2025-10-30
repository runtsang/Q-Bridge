from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import RealAmplitudes

class SharedClassName:
    """Hybrid estimator that unifies quantum circuit evaluation with classical‑inspired utilities."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # Fraud detection quantum builder -----------------------------------------
    @staticmethod
    def build_fraud_detection_program(
        input_params: "FraudLayerParameters",
        layers: Iterable["FraudLayerParameters"],
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        _apply_layer(circuit, input_params, clip=False)
        for layer in layers:
            _apply_layer(circuit, layer, clip=True)
        return circuit

    # Convolution filter quantum builder -------------------------------------
    @staticmethod
    def conv(kernel_size: int = 2, shots: int = 100, threshold: float = 127) -> QuantumCircuit:
        n_qubits = kernel_size ** 2
        circuit = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circuit.rx(theta[i], i)
        circuit.barrier()
        circuit += random_circuit(n_qubits, 2)
        circuit.measure_all()
        circuit._theta = theta
        circuit._shots = shots
        circuit._threshold = threshold
        return circuit

    # Autoencoder quantum builder --------------------------------------------
    @staticmethod
    def autoencoder(num_latent: int = 3, num_trash: int = 2) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit

# Helper functions for quantum fraud detection
def _apply_layer(circuit: QuantumCircuit, params: "FraudLayerParameters", *, clip: bool) -> None:
    # Simple two‑qubit mixing using CX
    circuit.cx(0, 1)
    # Parameterized rotations derived from the photonic parameters
    for i, theta in enumerate([params.bs_theta, params.bs_phi]):
        circuit.rx(theta if not clip else _clip(theta, 5), i)
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rx(r if not clip else _clip(r, 5), i)
        circuit.rz(phi, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(r if not clip else _clip(r, 5), i)
        circuit.rz(phi, i)
    for i, k in enumerate(params.kerr):
        circuit.rz(k if not clip else _clip(k, 1), i)

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# Dataclass for fraud detection parameters
from dataclasses import dataclass

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
