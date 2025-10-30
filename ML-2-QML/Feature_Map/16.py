"""ZZFeatureMapRZZControlled: a controlled‑modification of the RZZ feature map.

The original ZZFeatureMapRZZ uses a fixed pairwise coupling based on
(π−x)(π−y).  This variant replaces the coupling with a *data‑aware*
function that scales with the squared difference of the two features.
A global parameter ``pair_scale`` allows the user to tune the overall
interaction strength.  The single‑qubit rotations and entanglement
topology remain unchanged, preserving recognisability while
introducing richer, data‑dependent interactions.

Supported features
------------------
- Automatic resolution of common entanglement patterns: ``full``,
  ``linear``, ``circular`` or an explicit list of pairs.
- Optional user‑supplied data mapping for single‑qubit rotations.
- Optional user‑supplied coupling map for pairwise interactions.
- Normalisation toggle to keep the interaction angle within a
  user‑friendly range.
- Barrier insertion for debugging or visualisation.

Typical usage
-------------
>>> from zz_feature_map_rzz_controlled import zz_feature_map_rzz_controlled
>>> qc = zz_feature_map_rzz_controlled(4, reps=2, entanglement='circular')
"""

from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Return a list of qubit pairs according to the specification.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours
      - "circular": linear + (n-1, 0) if n > 2
      - explicit list of pairs
      - callable: f(n) -> list of pairs

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair refers to an
        out‑of‑range qubit.
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
        return [(int(i), int(j)) for i, j in pairs]

    # sequence of pairs
    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Feature map implementation
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    coupling_map_func: Callable[[ParameterExpression, ParameterExpression], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    normalize_interaction: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Create a ZZ‑feature map with RZZ entanglers whose angles are data‑dependent.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement pattern.
    data_map_func : callable | None, default None
        Mapping from a list of feature parameters to an angle for a
        single‑qubit rotation.  ``None`` uses the identity mapping
        (``φ1(x) = x``).
    coupling_map_func : callable | None, default None
        Mapping from two feature parameters to an angle for the RZZ
        entangler.  ``None`` uses the default data‑aware function
        ``(x_i - x_j)**2`` which captures the squared difference
        between the two features.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Insert barriers after each major block for easier visualisation.
    pair_scale : float, default 1.0
        Global scaling factor applied to every pairwise interaction.
    normalize_interaction : bool, default False
        If ``True``, the pairwise angle is divided by ``π**2`` to keep
        it in the range [0, 1] for typical feature values in [0, π].
    name : str | None, default None
        Name of the resulting circuit.  If ``None`` a default name is
        used.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(pair_scale, (int, float)):
        raise TypeError("pair_scale must be a numeric type.")
    if not isinstance(normalize_interaction, bool):
        raise TypeError("normalize_interaction must be a bool.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    x = ParameterVector(parameter_prefix, n)

    # Single‑qubit mapping
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

    # Pairwise coupling mapping
    if coupling_map_func is None:
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            angle = (xi - xj) ** 2
            if normalize_interaction:
                angle = angle / (pi ** 2)
            return angle
    else:
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            angle = coupling_map_func(xi, xj)
            if normalize_interaction:
                angle = angle / (pi ** 2)
            return angle

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified RZZ feature map.

    Parameters are identical to :func:`zz_feature_map_rzz_controlled`.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        coupling_map_func: Callable[[ParameterExpression, ParameterExpression], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        normalize_interaction: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            coupling_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            normalize_interaction,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
