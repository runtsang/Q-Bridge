"""ZZFeatureMapRZZControlled: A controlled‑modification variant of the original RZZ entangler feature map."""
from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# --------------------------------------------------------------------------- #
# 1. Utility functions
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Specification of the entanglement pattern.

        * ``"full"``   – all-to-all pairs (i < j).
        * ``"linear"`` – nearest‑neighbour pairs (0,1), (1,2), ….
        * ``"circular"`` – linear plus the wrap‑around pair (n-1,0) when n>2.
        * explicit list of tuples – e.g. ``[(0, 2), (1, 3)]``.
        * callable – ``f(num_qubits) -> iterable of (i, j)``.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of two‑qubit pairs.

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid indices.
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

    # Sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """
    Default single‑feature mapping φ₁(x) = x.
    """
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """
    Default two‑feature mapping φ₂(x, y) = x * y.
    """
    return x * y


def _default_pair_weight(i: int, j: int, n: int) -> float:
    """
    Default pair weight: uniform weight of 1.0 for all pairs.
    """
    return 1.0


# --------------------------------------------------------------------------- #
# 2. Feature‑map builder
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    pair_weight_func: Callable[[int, int, int], float] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    normalise: bool = True,
    shared_params: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a symmetric, distance‑weighted RZZ‑based feature map.

    This variant modifies the original RZZ entangler by:
    * introducing a weight function for each qubit pair,
    * normalising the interaction angles,
    * optionally sharing parameters across repetitions,
    * and supporting custom data‑mapping functions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features; must be >= 2.
    reps : int, default 2
        Number of repetition layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        Entanglement pattern specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        User‑supplied mapping from raw parameters to rotation angles. If ``None``, the defaults φ₁(x)=x and φ₂(x,y)=x*y are used.
    pair_weight_func : Callable[[int, int, int], float] | None, default None
        Function returning a weight for a pair (i,j) given the total qubit count ``n``. If ``None``, a uniform weight of 1.0 is used.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks for easier debugging.
    pair_scale : float, default 1.0
        Global scaling factor applied to all pairwise RZZ angles.
    normalise : bool, default True
        If ``True``, the pairwise angles are divided by the maximum pair weight to keep the circuit within a bounded range.
    shared_params : bool, default False
        If ``True``, the same ParameterVector is reused across all repetitions; otherwise each repetition gets its own vector.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If *feature_dimension* is less than 2 or if the entanglement specification is invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Resolve entanglement pairs once
    pairs = _resolve_entanglement(n, entanglement)

    # Determine weight function
    weight_func = pair_weight_func or _default_pair_weight

    # Compute maximum pair weight for normalisation
    max_weight = max(weight_func(i, j, n) for (i, j) in pairs) or 1.0

    # Prepare parameter vectors
    if shared_params:
        x_vectors = [ParameterVector(parameter_prefix, n)]
    else:
        x_vectors = [
            ParameterVector(f"{parameter_prefix}_{rep}", n) for rep in range(reps)
        ]

    # Default mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        map1 = lambda xi: data_map_func([xi])
        map2 = lambda xi, xj: data_map_func([xi, xj])

    # Build layers
    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit rotations
        for i in range(n):
            qc.p(2 * map1(x_vectors[rep][i]), i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit RZZ entanglers
        for (i, j) in pairs:
            weight = weight_func(i, j, n)
            angle = pair_scale * weight * map2(x_vectors[rep][i], x_vectors[rep][j])
            if normalise:
                angle = angle / max_weight
            qc.rzz(2 * angle, i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach the parameter vector(s) to the circuit for external binding
    qc.input_params = x_vectors[0] if shared_params else x_vectors
    return qc


# --------------------------------------------------------------------------- #
# 3. OO wrapper
# --------------------------------------------------------------------------- #
class ZZFeatureMapRZZControlled(QuantumCircuit):
    """
    OO interface for the RZZ‑based feature map with controlled modifications.

    Parameters are identical to :func:`zz_feature_map_rzz_controlled`.  The
    constructor simply builds the underlying circuit and composes it.

    Example
    -------
    >>> from zz_feature_map_rzz_controlled import ZZFeatureMapRZZControlled
    >>> qc = ZZFeatureMapRZZControlled(feature_dimension=4, reps=3)
    >>> qc.draw()
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        pair_weight_func: Callable[[int, int, int], float] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        normalise: bool = True,
        shared_params: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            pair_weight_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            normalise,
            shared_params,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
