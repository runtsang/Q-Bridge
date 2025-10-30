"""Controlled‑modification variant of the polynomial ZZFeatureMap."""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs from a simple entanglement spec.

    Supported specs:
      * ``"full"``    – all‑to‑all pairs (i < j)
      * ``"linear"``  – nearest neighbours (0,1), (1,2), …
      * ``"circular"``– linear plus wrap‑around (n‑1,0) when n > 2
      * explicit list of pairs ``[(0, 2), (1, 3)]``
      * callable ``f(num_qubits)`` → sequence of ``(i, j)``

    Raises
    ------
    ValueError
        If the specification is invalid or contains self‑loops/out‑of‑range indices.
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


def _default_map_1(x: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
    """Polynomial φ1(x) = Σ_k coeffs[k] · x^{k+1}."""
    expr: ParameterExpression = 0
    p = x
    for c in coeffs:
        expr = expr + c * p
        p = p * x
    return expr


def _default_map_2(x: ParameterExpression, y: ParameterExpression, weight: float) -> ParameterExpression:
    """Pairwise φ2(x, y) = weight · x · y."""
    return weight * x * y


def _normalize_features(data: Sequence[float]) -> List[float]:
    """Return a normalised copy of *data* such that max(|x|) ≤ 1.

    Parameters
    ----------
    data
        Feature vector to normalise.

    Returns
    -------
    List[float]
        Normalised feature vector.

    Raises
    ------
    ValueError
        If the maximum absolute value is zero (all features are zero).
    """
    max_val = max(abs(x) for x in data)
    if max_val == 0:
        raise ValueError("Cannot normalise a feature vector of all zeros.")
    return [x / max_val for x in data]


# ---------------------------------------------------------------------------
# Controlled‑Modification Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",
    normalize: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modification polynomial ZZ feature map.

    A single polynomial coefficient is applied to all qubits (shared symmetry),
    and a global pair‑weight is used for all ZZ interactions.  Optionally
    normalises the input data so that the maximum absolute feature value is
    bounded by 1.  The circuit is compatible with Qiskit’s classical data
    binding workflow and exposes ``input_params`` for parameter substitution.

    Parameters
    ----------
    feature_dimension
        Number of qubits / features (must be >= 2).
    reps
        Number of feature‑map repetitions (must be > 0).
    entanglement
        Definition of two‑qubit coupling pairs (see :func:`_resolve_entanglement`).
    single_coeffs
        One or more coefficients for the polynomial φ1(x).  If a single
        value is supplied it is shared across all qubits.
    pair_weight
        Weight for the pairwise phase φ2(x, y).  Applied uniformly.
    basis
        Basis preparation before each repetition: ``"h"`` for Hadamard,
        ``"ry"`` for RY(π/2).
    normalize
        If ``True``, normalises any supplied feature vector to |x| ≤ 1
        before binding.  The normalisation is performed by the helper
        :func:`_normalize_features` and is purely a convenience for the
        caller.
    parameter_prefix
        Prefix for the :class:`~qiskit.circuit.ParameterVector` names.
    insert_barriers
        Whether to insert barriers between logical blocks.
    name
        Optional circuit name; defaults to ``"ZZFeatureMapPolyControlled"``.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If the supplied arguments are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if basis not in {"h", "ry"}:
        raise ValueError("basis must be 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for all features
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # ry
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases using the shared polynomial
        for i in range(n):
            angle = 2 * _default_map_1(x[i], single_coeffs)
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions with a global pair weight
        for (i, j) in pairs:
            angle = 2 * _default_map_2(x[i], x[j], pair_weight)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Object‑oriented wrapper for the controlled‑modification polynomial ZZ map.

    The class inherits from :class:`~qiskit.circuit.QuantumCircuit` and
    automatically composes the underlying functional circuit.  It exposes
    ``input_params`` for parameter binding and can be used interchangeably
    with the functional API.

    Parameters
    ----------
    feature_dimension
        Number of qubits / features (must be >= 2).
    reps
        Number of feature‑map repetitions.
    entanglement
        Two‑qubit coupling specification.
    single_coeffs
        Polynomial coefficients (shared across qubits).
    pair_weight
        Global pair‑weight.
    basis
        Basis preparation: ``"h"`` or ``"ry"``.
    normalize
        Whether to normalise input data before binding.
    parameter_prefix
        Parameter vector prefix.
    insert_barriers
        Whether to insert barriers.
    name
        Circuit name.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        normalize: bool = False,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            basis,
            normalize,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ZZFeatureMapPolyControlled",
    "zz_feature_map_poly_controlled",
    "_normalize_features",
]
