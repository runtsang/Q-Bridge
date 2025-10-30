"""ZZFeatureMapRZZControlled variant with symmetric RZZ interactions and data‑dependent scaling."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on the entanglement specification.

    Supported specifications:
      * ``"full"``   – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest neighbours (0,1), (1,2), …
      * ``"circular"`` – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs, e.g. ``[(0, 2), (1, 3)]``
      * callable ``f(num_qubits) -> sequence of (i, j)``

    Raises
    ------
    ValueError
        If an unknown string is supplied or pairs are invalid.
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
# Controlled‑modification variant
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float | ParameterExpression = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a ZZ feature map with symmetric RZZ entanglement and data‑dependent scaling.

    The pairwise rotation angle is
    ``2 * pair_scale * φ2(xi, xj) * (xi + xj)``.
    This introduces a symmetric dependence on both features and a tunable
    pair‑scale that can be a constant or a Qiskit ParameterExpression.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be at least 2).
    reps : int, default 2
        Number of repetitions of the circuit layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional custom mapping from raw feature parameters to rotation angles.
        If None, defaults to φ1 and φ2 defined above.
    parameter_prefix : str, default "x"
        Prefix for the generated ParameterVector.
    insert_barriers : bool, default False
        Insert barriers between layers for easier visualisation.
    pair_scale : float | ParameterExpression, default 1.0
        Global scaling factor applied to all RZZ interactions. Can be a
        Qiskit ParameterExpression for dynamic control.
    name : str | None, default None
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        A parameterised feature‑map circuit ready for binding.

    Raises
    ------
    ValueError
        If input parameters are out of bounds or invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for input features
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Build layers
    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single‑qubit rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        # Two‑qubit RZZ rotations with data‑dependent scaling
        for (i, j) in pairs:
            # Data‑dependent scaling: (xi + xj)
            scaling_factor = x[i] + x[j]
            angle = 2 * pair_scale * map2(x[i], x[j]) * scaling_factor
            qc.rzz(angle, i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Store the input parameters for binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification RZZ feature map.

    Attributes
    ----------
    input_params : ParameterVector
        The vector of parameters that must be bound before execution.

    Notes
    -----
    The wrapper simply composes the circuit built by :func:`zz_feature_map_rzz_controlled`.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float | ParameterExpression = 1.0,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension, reps, entanglement, data_map_func,
            parameter_prefix, insert_barriers, pair_scale, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
