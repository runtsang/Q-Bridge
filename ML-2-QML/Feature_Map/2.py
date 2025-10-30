"""ZZFeatureMapRZZControlled: Symmetric RZZ entanglement with shared pair‑scale.

This feature map is a controlled modification of the original ZZFeatureMapRZZ.
Key changes:
- All two‑qubit interactions use a single pair‑scale coefficient.
- The pair‑mapping function is shared across all pairs, reducing the number of
  distinct parameters and encouraging smoother optimisation landscapes.
- Optional data normalisation to the range [0, π] is available.
- The interface remains identical to the seed: feature_dimension, reps,
  entanglement, data_map_func, parameter_prefix, insert_barriers,
  pair_scale, normalise_data, name.

Supported entanglement patterns:
    * "full"   – all‑to‑all pairs
    * "linear" – nearest‑neighbour chain
    * "circular" – linear plus wrap‑around
    * explicit list of (i, j) tuples
    * callable f(num_qubits) → sequence of pairs

The circuit is fully Qiskit‑compatible and exposes a helper function
`zz_feature_map_rzz_controlled` and a class `ZZFeatureMapRZZControlled`
for functional or object‑oriented use.

"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Entanglement resolution utility
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full":     all-to-all pairs (i < j)
      - "linear":   nearest neighbours (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs, e.g. [(0, 2), (1, 3)]
      - callable:   f(num_qubits) → sequence of (i, j)
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


# ---------------------------------------------------------------------------
# Default mapping functions
# ---------------------------------------------------------------------------

def _default_phi1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit φ₁(x) = x."""
    return x


def _default_phi2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default pair‑wise φ₂(x, y) = (π - x)(π - y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a symmetrised ZZ feature map with RZZ entanglers.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits (must be >= 2).
    reps : int, default 2
        Number of repeated layers.
    entanglement : str | list | callable, default "full"
        Entanglement specification (see :func:`_resolve_entanglement`).
    data_map_func : callable, optional
        User‑supplied mapping function that takes a sequence of
        ParameterExpression objects and returns a single ParameterExpression.
        If None, default φ₁ and φ₂ are used.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Insert barriers between layers for easier visualisation.
    pair_scale : float, default 1.0
        Global scaling factor applied to all pair‑wise angles.
    normalise_data : bool, default False
        If True, normalise each input feature to the range [0, π] before
        applying the mapping functions.
    name : str, optional
        Circuit name. If None, defaults to ``"ZZFeatureMapRZZControlled"``.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``reps`` < 1, or ``pair_scale`` <= 0.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_scale <= 0.0:
        raise ValueError("pair_scale must be positive.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for each feature
    x = ParameterVector(parameter_prefix, n)

    # Define single‑qubit and pair‑wise map functions
    if data_map_func is None:
        def phi1(xi: ParameterExpression) -> ParameterExpression:
            return _default_phi1(xi)
        def phi2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return _default_phi2(xi, xj)
    else:
        def phi1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def phi2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Normalisation wrapper
    if normalise_data:
        def norm(xi: ParameterExpression) -> ParameterExpression:
            return pi * xi
        def phi1_norm(xi: ParameterExpression) -> ParameterExpression:
            return phi1(norm(xi))
        def phi2_norm(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return phi2(norm(xi), norm(xj))
        phi1, phi2 = phi1_norm, phi2_norm

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * phi1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * phi2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Object‑oriented wrapper for :func:`zz_feature_map_rzz_controlled`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        normalise_data: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            normalise_data,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
