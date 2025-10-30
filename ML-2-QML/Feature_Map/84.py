"""ZZFeatureMapRZZControlled: Controlled modification of the RZZ entangler variant."""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest‑neighbor pairs (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs
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
    else:
        pairs = list(entanglement)

    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    shared_pair_scale: float = 1.0,
    normalise: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modification variant of the ZZ feature map using native ``rzz`` gates.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default: 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default: "full"
        Defines the two‑qubit coupling pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default: None
        Function mapping a list of feature parameters to a single rotation angle.
        If ``None``, defaults to ``φ1(x)=x`` for single‑qubit terms and
        ``φ2(x, y)=(π−x)(π−y)`` for two‑qubit terms.
    parameter_prefix : str, default: "x"
        Prefix for the automatically generated parameter vector.
    insert_barriers : bool, default: False
        Insert barriers between layers for easier debugging.
    shared_pair_scale : float, default: 1.0
        Global scaling factor applied to all RZZ interactions.  When ``normalise`` is
        ``True`` the effective scale becomes ``shared_pair_scale / sqrt(num_pairs)``.
    normalise : bool, default: True
        If ``True`` normalises the pair‑scaling to keep rotation angles bounded
        when the number of pairs grows large.
    name : str | None, default: None
        Quantum circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with data vectors.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``reps`` < 1, or if the entanglement
        specification contains invalid pairs.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")
    x = ParameterVector(parameter_prefix, n)

    # Default data mapping if none supplied
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
    num_pairs = len(pairs)

    # Effective pair‑scaling
    scale_factor = shared_pair_scale
    if normalise and num_pairs > 0:
        scale_factor = shared_pair_scale / (num_pairs ** 0.5)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * scale_factor * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for :func:`zz_feature_map_rzz_controlled`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        shared_pair_scale: float = 1.0,
        normalise: bool = True,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            shared_pair_scale,
            normalise,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
