from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of unique qubit pairs.

    Supported specs:
      * ``"full"``   – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest neighbours (0,1), (1,2), …
      * ``"circular"`` – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of tuples ``[(0, 2), (1, 3)]``
      * callable ``f(num_qubits) -> sequence of (i, j)``
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_all(xs: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default φₙ(x₀, …, x_{n-1}) = ∏ (π − xᵢ)."""
    prod = 1
    for x in xs:
        prod *= (pi - x)
    return prod


def _apply_pre_rotation(
    qc: QuantumCircuit,
    rotation: str,
    qubits: Sequence[int],
) -> None:
    """Apply a single‑qubit pre‑rotation to each qubit."""
    if rotation == "rx":
        qc.rx(pi / 4, qubits)
    elif rotation == "ry":
        qc.ry(pi / 4, qubits)
    elif rotation == "rz":
        qc.rz(pi / 4, qubits)
    else:
        raise ValueError(f"Unsupported pre‑rotation type: {rotation!r}")


def _apply_post_rotation(
    qc: QuantumCircuit,
    rotation: str,
    qubits: Sequence[int],
) -> None:
    """Apply a single‑qubit post‑rotation to each qubit."""
    if rotation == "rx":
        qc.rx(pi / 4, qubits)
    elif rotation == "ry":
        qc.ry(pi / 4, qubits)
    elif rotation == "rz":
        qc.rz(pi / 4, qubits)
    else:
        raise ValueError(f"Unsupported post‑rotation type: {rotation!r}")


def _apply_global_interaction(
    qc: QuantumCircuit,
    angle: ParameterExpression,
    qubits: Sequence[int],
) -> None:
    """Apply a global RZ rotation encoding higher‑order interactions."""
    qc.rz(angle, qubits)
