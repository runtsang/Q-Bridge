"""Quantum‑centric factory that can build a variational circuit or a quantum‑enhanced transformer.\n\nThe module keeps the original `build_classifier_circuit` signature but\nextends it with a `use_transformer` flag.  When `False` a simple data‑uploading\nansatz is created with Qiskit.  When `True` a placeholder transformer is\nreturned; the full quantum transformer will be implemented in a future\nrelease.\n"""  

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# 1.  Variational circuit (data‑uploading ansatz)
# --------------------------------------------------------------------------- #
def _build_variational_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered variational circuit identical to the original seed."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(encoding[i], i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 2.  Quantum‑enhanced transformer (placeholder)
# --------------------------------------------------------------------------- #
class _QuantumTransformer(tq.QuantumModule):
    """Placeholder quantum transformer.  It simply forwards the input through a\nclassic transformer implemented with TorchQuantum.  The full quantum\nattention/FFN are omitted for brevity."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # We keep the classic transformer inside a QuantumModule for API parity.
        self.transformer = tq.QuantumDevice  # dummy placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Full quantum transformer not implemented in this stub.")


# --------------------------------------------------------------------------- #
# 3.  Public factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    use_transformer: bool = False,
    num_heads: int = 4,
    ffn_dim: int = 128,
    dropout: float = 0.1,
) -> Tuple[QuantumCircuit | _QuantumTransformer, Iterable[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Return (circuit, encoding, weights, observables).  For `use_transformer=True`\n    the function currently raises `NotImplementedError`.  The non‑transformer\n    branch builds a standard variational circuit.\n    """
    if use_transformer:
        raise NotImplementedError("Quantum transformer is not yet implemented in this module.")
    return _build_variational_circuit(num_qubits, depth)


__all__ = ["build_classifier_circuit"]
