"""zz_feature_map_rzz_extended: an extended version of the original RZZ‑based feature map.

This module defines:
- ``zz_feature_map_rzz_extended`` – a functional helper returning a ``QuantumCircuit``.
- ``ZZFeatureMapRZZExtended`` – an OO wrapper subclassing ``QuantumCircuit``.
Both share the same signature and parameter semantics.

The extension adds:
* optional pre‑ and post‑single‑qubit rotations (`pre_rotations`, `post_rotations`);
* a configurable interaction order (`order`) that adds an extra entanglement layer;
* robust validation and clear error messages for parameter bounds.

The circuit is compatible with Qiskit’s data‑binding workflow and can be used directly in variational algorithms or as a feature map for quantum kernel estimation.
"""

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
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbors (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
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


# ---------------------------------------------------------------------------
# Functional implementation
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    order: int = 2,
    pre_rotations: bool = False,
    post_rotations: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended RZZ feature map with optional pre/post rotations and higher‑order entanglement.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, default 2
        Number of repetition layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of pairwise entanglement.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping from raw features to rotation angles.
    parameter_prefix : str, default "x"
        Prefix for generated parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    pair_scale : float, default 1.0
        Scaling factor for pair entanglement angles.
    order : int, default 2
        Interaction order. 2 corresponds to pairwise RZZ only; >2 adds an extra entanglement
        layer with a circular pattern.
    pre_rotations : bool, default False
        If True, apply a single‑qubit P rotation before the H gates in each layer.
    post_rotations : bool, default False
        If True, apply a single‑qubit P rotation after all entanglement in each layer.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If input parameters are out of bounds.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if order < 2:
        raise ValueError("order must be >= 2.")
    if pair_scale <= 0:
        raise ValueError("pair_scale must be positive.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtended")

    # Parameter vector for all features
    x = ParameterVector(parameter_prefix, n)

    # Default mapping functions if none provided
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

    # Resolve pairwise entanglement
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotations
        if pre_rotations:
            for i in range(n):
                qc.p(map1(x[i]), i)

        qc.h(range(n))

        if insert_barriers:
            qc.barrier()

        # Phase rotations after H
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise RZZ entanglement
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)

        # Higher‑order entanglement if requested
        if order > 2:
            extra_pairs = _resolve_entanglement(n, "circular")
            for (i, j) in extra_pairs:
                qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)

        # Optional post‑rotations
        if post_rotations:
            for i in range(n):
                qc.p(map1(x[i]), i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Object‑oriented wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapRZZExtended(QuantumCircuit):
    """
    OO wrapper for the extended RZZ feature map.

    Parameters are identical to :func:`zz_feature_map_rzz_extended`.  Instantiating
    this class yields a fully composed ``QuantumCircuit`` with the same
    parameter vector exposed as ``input_params``.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        order: int = 2,
        pre_rotations: bool = False,
        post_rotations: bool = False,
        name: str = "ZZFeatureMapRZZExtended",
    ) -> None:
        built = zz_feature_map_rzz_extended(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            order,
            pre_rotations,
            post_rotations,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtended", "zz_feature_map_rzz_extended"]
