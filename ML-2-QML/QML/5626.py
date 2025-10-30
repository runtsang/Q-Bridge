"""
Quantum counterpart of :class:`HybridClassifierModel`.

The circuit integrates:
1. Feature encoding via RX gates.
2. A variational RealAmplitudes ansatz.
3. A swap‑test auto‑encoder block.
4. A photonic‑style U3 variational layer.

The module exposes the same ``build_classifier_circuit`` API.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli


class HybridClassifierModel:
    """
    Quantum implementation of the hybrid classifier.  The forward pass
    evaluates expectation values of Z observables on each qubit.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = build_classifier_circuit(num_qubits, depth)
        self.simulator = Aer.get_backend("statevector_simulator")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for an input vector ``x`` and return the
        expectation values of the observables.
        """
        bound_circuit = self.circuit.bind_parameters(dict(zip(self.encoding, x)))
        job = execute(bound_circuit, self.simulator)
        state_vec = job.result().get_statevector(bound_circuit)
        state = Statevector(state_vec)
        return np.array([state.expectation_value(obs) for obs in self.observables])


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a hybrid quantum circuit with encoding, ansatz, swap‑test
    auto‑encoder and photonic‑style variational layer.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Feature encoding
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    # Variational ansatz (RealAmplitudes style)
    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Swap‑test auto‑encoder block
    aux_reg = QuantumRegister(1, "aux")
    qc.add_register(aux_reg)
    qc.h(aux_reg[0])
    for i in range(num_qubits):
        qc.cswap(aux_reg[0], i, i)
    qc.h(aux_reg[0])

    # Photonic‑style variational layer (U3 as a stand‑in)
    for i in range(num_qubits):
        qc.u3(0.5, 0.3, 0.7, i)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp.from_label("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, encoding, weights, observables


__all__ = ["HybridClassifierModel", "build_classifier_circuit"]
