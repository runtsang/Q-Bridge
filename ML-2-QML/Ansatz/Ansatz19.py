"""RealAmplitudes variant with alternating RY/RX rotation layers and depth‑controlled expansion."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on a simple entanglement spec.
    Parameters
    ----------
    *   ``num_qubits``: number of qubits.
    *   ``entanglement``: pre‑defined string or custom pair list.
    *   ``entric**…"""
    # (body omitted for brevity)...
