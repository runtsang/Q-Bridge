"""Quantum classifier circuit mirroring the classical helper.

The circuit uses RX rotations for data encoding, Ry rotations for variational parameters,
CZ gates for entanglement, and optional CRX gates to emulate self‑attention.
A transformer‑style repetition of rotation‑entanglement blocks is supported.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    transformer_layers: int = 1,
    crx_entangle: bool = True,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (identical to the feature dimension of the classical model).
    depth : int
        Depth of the variational ansatz – number of Ry layers.
    transformer_layers : int, optional
        Additional transformer‑style layers that repeat a rotation followed by
        CRX entanglement (self‑attention style).
    crx_entangle : bool, optional
        If True, CRX gates are used for self‑attention entanglement; otherwise
        CZ gates are used.

    Returns
    -------
    QuantumCircuit
        The assembled circuit.
    Iterable
        List of encoding parameters (RX angles).
    Iterable
        List of variational parameters (Ry and optional CRX angles).
    List[SparsePauliOp]
        Observables – single‑qubit Z measurements for each qubit.
    """
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters – Ry for each depth layer plus CRX angles per transformer layer
    ry_params = ParameterVector("ry", num_qubits * depth)
    crx_params = ParameterVector("crx", num_qubits * transformer_layers) if crx_entangle else ParameterVector("crx", 0)

    # Main circuit
    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    # Variational layers
    ry_idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(ry_params[ry_idx], i)
            ry_idx += 1
        # Entanglement – CZ across nearest neighbours
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Transformer‑style self‑attention layers
    crx_idx = 0
    for _ in range(transformer_layers):
        # Rotate all qubits again
        for i in range(num_qubits):
            qc.rx(ry_params[ry_idx], i)
            ry_idx += 1
        # Self‑attention entanglement
        if crx_entangle:
            for i in range(num_qubits - 1):
                qc.crx(crx_params[crx_idx], i, i + 1)
                crx_idx += 1
            qc.crx(crx_params[crx_idx], num_qubits - 1, 0)
            crx_idx += 1
        else:
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)

    # Observables – Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    # Flatten parameter vectors for return
    all_params = list(encoding) + list(ry_params) + list(crx_params)

    return qc, list(encoding), all_params, observables


__all__ = ["build_classifier_circuit"]
