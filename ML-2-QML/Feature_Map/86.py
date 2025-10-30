"""ZZFeatureMapRZZExtended: an extended RZZ‑entanglement feature map with higher‑order interactions and adaptive depth.

The module defines:
- ``zz_feature_map_rzz_extended``: functional API
- ``ZZFeatureMapRZZExtended``: QuantumCircuit subclass

Features
--------
* Adaptive repetition depth (int or list of two ints)
* Optional triplet interaction layer (RZ on one qubit)
* Optional data normalisation (scales features to [0, 2π])
* Parameterised data mapping function
* Custom entanglement specification for pairwise coupling
* Barrier insertion for debugging

Supported data types
--------------------
- Qiskit ParameterExpression
- Numeric values or ParameterVector

Examples
--------
>>> from qiskit import QuantumCircuit
>>> qc = zz_feature_map_rzz_extended(feature_dimension=4, reps=3, triplet_scale=0.8)
>>> qc.draw()
"""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n-1,0) if n > 2
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _resolve_triplet_entanglement(
    num_qubits: int,
    triplet_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Return a list of triplet indices for higher‑order interaction.

    Supported specs:
      - "full": all distinct triples (i<j<k)
      - "linear": (0,1,2), (1,2,3), …
      - "circular": linear plus wrap‑around (n-2,n-1,0)
      - explicit list of triples
      - callable: f(num_qubits) -> sequence of triples
    """
    if isinstance(triplet_entanglement, str):
        if triplet_entanglement == "full":
            return list(itertools.combinations(range(num_qubits), 3))
        if triplet_entanglement == "linear":
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        if triplet_entanglement == "circular":
            pairs = [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
            if num_qubits > 3:
                pairs.append((num_qubits - 2, num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown triplet_entanglement string: {triplet_entanglement!r}")

    if callable(triplet_entanglement):
        pairs = list(triplet_entanglement(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in pairs]

    # sequence of triples
    pairs = [(int(i), int(j), int(k)) for (i, j, k) in triplet_entanglement]  # type: ignore[arg-type]
    for (i, j, k) in pairs:
        if len({i, j, k})!= 3:
            raise ValueError("Triplet entanglement pairs must be distinct.")
        for idx in (i, j, k):
            if not (0 <= idx < num_qubits):
                raise ValueError(f"Triplet entanglement index {(i, j, k)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Feature‑map implementation
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_extended(
    feature_dimension: int,
    reps: int | List[int] = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    triplet_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    triplet_scale: float = 0.5,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Extended RZZ feature map with higher‑order interactions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int or list[int]
        Depth of the circuit.  If an int, the same depth is applied to both
        pairwise and triplet layers.  If a list, its length must be 2 and
        each element specifies the repetitions for the corresponding layer.
    entanglement : str or sequence or callable
        Specification of pairwise entanglement pairs.
    triplet_entanglement : str or sequence or callable
        Specification of triples for the higher‑order interaction layer.
    data_map_func : callable or None
        Optional function mapping a list of ParameterExpressions to a
        single ParameterExpression.  If None, the default φ1, φ2, φ3
        functions are used.
    parameter_prefix : str
        Prefix for the ParameterVector.
    insert_barriers : bool
        Insert barriers between logical blocks for easier debugging.
    pair_scale : float
        Scaling factor for the pairwise RZZ angles.
    triplet_scale : float
        Scaling factor for the triplet RZ angles.
    normalize : bool
        If True, features are multiplied by 2π before mapping.
    name : str or None
        Name for the created circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    # Resolve repetition depths
    if isinstance(reps, int):
        reps_list = [reps, reps]
    elif isinstance(reps, list):
        if len(reps)!= 2:
            raise ValueError("reps list must have length 2 for pair and triplet layers.")
        reps_list = reps
    else:
        raise TypeError("reps must be int or list[int].")

    # Resolve entanglement patterns
    pairs = _resolve_entanglement(n, entanglement)
    triplets = _resolve_triplet_entanglement(n, triplet_entanglement)

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return _default_map_1(xi * (2 * pi if normalize else 1))
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return _default_map_2(xi * (2 * pi if normalize else 1), xj * (2 * pi if normalize else 1))
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return _default_map_3(xi * (2 * pi if normalize else 1), xj * (2 * pi if normalize else 1), xk * (2 * pi if normalize else 1))
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi * (2 * pi if normalize else 1)])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi * (2 * pi if normalize else 1), xj * (2 * pi if normalize else 1)])
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi * (2 * pi if normalize else 1), xj * (2 * pi if normalize else 1), xk * (2 * pi if normalize else 1)])

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtended")

    # Pairwise layer
    for _ in range(reps_list[0]):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)
        if insert_barriers:
            qc.barrier()

    # Triplet layer
    for _ in range(reps_list[1]):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for i, j, k in triplets:
            qc.rz(2 * triplet_scale * map3(x[i], x[j], x[k]), i)
        if insert_barriers:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZExtended(QuantumCircuit):
    """QuantumCircuit subclass for the extended RZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int or list[int]
        Layer depths.
    entanglement : str or sequence or callable
        Pairwise entanglement specification.
    triplet_entanglement : str or sequence or callable
        Triplet entanglement specification.
    data_map_func : callable or None
        Optional data mapping function.
    parameter_prefix : str
        Prefix for parameters.
    insert_barriers : bool
        Insert barriers.
    pair_scale : float
        Pairwise scaling factor.
    triplet_scale : float
        Triplet scaling factor.
    normalize : bool
        Normalise features.
    name : str
        Name of the circuit.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int | List[int] = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        triplet_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        triplet_scale: float = 0.5,
        normalize: bool = False,
        name: str = "ZZFeatureMapRZZExtended",
    ) -> None:
        built = zz_feature_map_rzz_extended(
            feature_dimension,
            reps,
            entanglement,
            triplet_entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            triplet_scale,
            normalize,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtended", "zz_feature_map_rzz_extended"]
