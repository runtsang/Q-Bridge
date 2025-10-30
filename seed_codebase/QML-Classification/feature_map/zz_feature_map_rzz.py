"""ZZFeatureMap variant using native ``rzz`` gates for pairwise entanglement."""
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
    """Return a list of two-qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), ...
      - "circular": linear plus wrap-around (n-1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) -> sequence of (i, j)
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
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)

# ---------------------------------------------------------------------------
# Variant 1: RZZ entanglers (native two-qubit rotation)
# ---------------------------------------------------------------------------

def zz_feature_map_rzz(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """ZZ feature map variant using native `rzz` for pair coupling.

    Pair angle is 2·pair_scale·φ2(xi, xj). Defaults to φ1(x)=x, φ2(x,y)=(π−x)(π−y).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZ")

    x = ParameterVector(parameter_prefix, n)

    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - xi) * (pi - xj)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)
        if insert_barriers and rep != reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZ(QuantumCircuit):
    """Class-style wrapper for the RZZ-entangler variant."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        name: str = "ZZFeatureMapRZZ",
    ) -> None:
        built = zz_feature_map_rzz(
            feature_dimension, reps, entanglement, data_map_func,
            parameter_prefix, insert_barriers, pair_scale, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZ", "zz_feature_map_rzz"]
