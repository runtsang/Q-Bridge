"""Extended ZZ Feature Map with RZZ entanglers and higher‑order terms.

This module defines a quantum circuit that encodes classical data into a
parameterised Ansatz consisting of:
- 1‑qubit rotations (φ1(x))
- 2‑qubit RZZ entanglement (φ2(xi, xj))
- 3‑qubit controlled‑phase entanglement (φ3(xi, xj, xk))

The circuit supports configurable depth, optional normalisation, and
pre/post rotations.  It can be instantiated either via the functional
helper `zz_feature_map_rzz_extended` or the OO wrapper `ZZFeatureMapRZZExtended`."""
from __future__ import annotations

import itertools
from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
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


def _resolve_triples(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Return a list of three‑qubit triples according to a simple spec.

    The spec semantics are identical to _resolve_entanglement but for triples.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return list(itertools.combinations(range(num_qubits), 3))
        if entanglement == "linear":
            # linear triples: (i, i+1, i+2)
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        if entanglement == "circular":
            triples = [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
            if num_qubits > 3:
                triples.append((num_qubits - 2, num_qubits - 1, 0))
                triples.append((num_qubits - 1, 0, 1))
            return triples
        raise ValueError(f"Unknown triple entanglement string: {entanglement!r}")

    if callable(entanglement):
        triples = list(entanglement(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in triples]

    triples = [(int(i), int(j), int(k)) for (i, j, k) in entanglement]  # type: ignore[arg-type]
    for (i, j, k) in triples:
        if len({i, j, k})!= 3:
            raise ValueError("Triple entanglement must involve distinct qubits.")
        if not all(0 <= idx < num_qubits for idx in (i, j, k)):
            raise ValueError(f"Triple entanglement {(i, j, k)} out of range for n={num_qubits}.")
    return triples


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
# Functional feature map
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    triple_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    triple_scale: float = 1.0,
    normalise: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    interaction_depth: int = 1,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended ZZ feature map using RZZ for pairwise entanglement and ccphase for
    three‑qubit interactions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int, default 2
        Number of repetition cycles.
    entanglement : str or sequence or callable, default "full"
        Specification of pairwise entanglement pairs.
    triple_entanglement : str or sequence or callable, default "full"
        Specification of three‑qubit entanglement triples.
    data_map_func : callable, optional
        Function mapping a list of ParameterExpression to a single
        ParameterExpression.  If provided, it is used for all mapping
        functions (φ1, φ2, φ3).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    pair_scale : float, default 1.0
        Scaling factor applied to pairwise RZZ angles.
    triple_scale : float, default 1.0
        Scaling factor applied to triple‑qubit ccphase angles.
    normalise : bool, default False
        If True, all input parameters are multiplied by π before mapping,
        effectively scaling raw features into the [0, π] range.
    pre_rotation : bool, default False
        If True, a π/2 phase rotation is applied to all qubits before each
        pairwise entanglement layer.
    post_rotation : bool, default False
        If True, a π/2 phase rotation is applied after each triple‑qubit
        interaction layer.
    interaction_depth : int, default 1
        Number of nested pair/triple layers per repetition.  Higher values
        increase circuit depth linearly.
    name : str, optional
        Custom circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for embedding into a variational
        algorithm or as a data encoder.

    Raises
    ------
    ValueError
        If ``feature_dimension`` is less than 2 (or 3 when triple
        interactions are requested).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    if triple_scale!= 0.0 or interaction_depth > 1:
        if n < 3:
            raise ValueError("Triple‑qubit interactions require at least 3 qubits.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtended")

    x = ParameterVector(parameter_prefix, n)

    # Define mapping functions, applying normalisation if requested.
    if data_map_func is None:
        map1 = lambda xi: _default_map_1(xi) * (pi if normalise else 1)
        map2 = lambda xi, xj: _default_map_2(xi, xj) * (pi if normalise else 1)
        map3 = lambda xi, xj, xk: _default_map_3(xi, xj, xk) * (pi if normalise else 1)
    else:
        map1 = lambda xi: data_map_func([xi]) * (pi if normalise else 1)
        map2 = lambda xi, xj: data_map_func([xi, xj]) * (pi if normalise else 1)
        map3 = lambda xi, xj, xk: data_map_func([xi, xj, xk]) * (pi if normalise else 1)

    pairs = _resolve_entanglement(n, entanglement)
    triples = _resolve_triples(n, triple_entanglement)

    layers = reps * interaction_depth

    for layer in range(layers):
        qc.h(range(n))
        if pre_rotation:
            for i in range(n):
                qc.p(pi / 2, i)

        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)

        for (i, j, k) in triples:
            qc.ccphase(2 * triple_scale * map3(x[i], x[j], x[k]), i, j, k)

        if post_rotation:
            for i in range(n):
                qc.p(pi / 2, i)

        if insert_barriers and layer!= layers - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc

# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapRZZExtended(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_rzz_extended`.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default 2
        Number of repetition cycles.
    entanglement : str or sequence or callable, default "full"
        Pairwise entanglement specification.
    triple_entanglement : str or sequence or callable, default "full"
        Triple‑qubit entanglement specification.
    data_map_func : callable, optional
        Custom mapping function.
    parameter_prefix : str, default "x"
        Prefix for ParameterVector.
    insert_barriers : bool, default False
        Insert barriers between layers.
    pair_scale : float, default 1.0
        Scaling for pairwise RZZ angles.
    triple_scale : float, default 1.0
        Scaling for triple‑qubit ccphase angles.
    normalise : bool, default False
        Normalise raw features into [0, π].
    pre_rotation : bool, default False
        Pre‑rotation before each pairwise layer.
    post_rotation : bool, default False
        Post‑rotation after each triple layer.
    interaction_depth : int, default 1
        Depth of nested pair/triple layers per repetition.
    name : str, default "ZZFeatureMapRZZExtended"
        Circuit name.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        triple_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        triple_scale: float = 1.0,
        normalise: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        interaction_depth: int = 1,
        name: str = "ZZFeatureMapRZZExtended",
    ) -> None:
        built = zz_feature_map_rzz_extended(
            feature_dimension,
            reps,
            entanglement,
            triple_entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            triple_scale,
            normalise,
            pre_rotation,
            post_rotation,
            interaction_depth,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapRZZExtended", "zz_feature_map_rzz_extended"]
