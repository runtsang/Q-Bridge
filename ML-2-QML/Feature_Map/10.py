"""Quantum feature map with symmetric pairwise interactions and odd‑degree polynomial single‑qubit terms."""
from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ---------------------------------------------------------------------------
# Utility: resolve entanglement specification
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Translate various entanglement specifications into a list of two‑qubit pairs.

    Supported specifications:
      * ``"full"``      – all‑to‑all pairs (i < j)
      * ``"linear"``    – nearest‑neighbour chain
      * ``"circular"``  – linear plus wrap‑around (n‑1,0)
      * explicit list of tuples ``[(i, j), …]``
      * callable ``f(n) -> Iterable[(i,j)]``

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement specification.

    Returns
    -------
    List[Tuple[int, int]]
        List of valid qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification is supplied or pairs are out of range.
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

    # explicit sequence
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs

# ---------------------------------------------------------------------------
# Helper functions for parameter maps
# ---------------------------------------------------------------------------

def _odd_degree_poly(x: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
    """
    Construct an odd‑degree polynomial with alternating sign pattern.

    φ(x) = Σ_k (-1)^k * coeffs[k] * x^{2k+1}

    Parameters
    ----------
    x : ParameterExpression
        Input parameter.
    coeffs : Sequence[float]
        Coefficients for the odd powers, starting with the first power.

    Returns
    -------
    ParameterExpression
        Polynomial expression.
    """
    expr: ParameterExpression = 0
    power: ParameterExpression = x  # x^{1}
    sign = 1
    for c in coeffs:
        expr += sign * c * power
        power = power * x * x  # advance to next odd power
        sign *= -1
    return expr

def _distance_weight(i: int, j: int, base: float = 1.0) -> float:
    """
    Default symmetric pair‑weight decaying with qubit distance.

    w(i,j) = base / (|i-j| + 1)

    Parameters
    ----------
    i, j : int
        Qubit indices.
    base : float
        Base weight.

    Returns
    -------
    float
        Weight for the pair (i, j).
    """
    distance = abs(i - j)
    return base / (distance + 1)

# ---------------------------------------------------------------------------
# Feature map construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0, 0.5),
    pair_weight: float | Callable[[int, int], float] = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalise: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a ZZ feature map with symmetric pairwise interactions and an odd‑degree
    single‑qubit polynomial map.

    The circuit is built in the following way:

    * Optional basis preparation (`Hadamard` or `RY(π/2)`).
    * Single‑qubit phase rotation `P(2 φ1(x_i))` for each qubit.
    * ZZ interaction via `CX – P(2 φ2(x_i,x_j)) – CX` for each entangled pair.

    Parameters
    ----------
    feature_dimension : int
        Number of data features / qubits.
    reps : int, default 2
        Number of repeated layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement pattern.
    single_coeffs : Sequence[float], default (1.0, 0.5)
        Coefficients for the odd‑degree polynomial. The first coefficient
        corresponds to the linear term; subsequent coefficients are for
        higher odd powers.
    pair_weight : float | Callable[[int, int], float], default 1.0
        Base weight for pair interactions. If a callable is supplied it
        receives the pair indices and returns a weight.
    basis : str, default "h"
        Basis preparation; either ``"h"`` for Hadamard or ``"ry"`` for
        RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for clarity.
    normalise : bool, default False
        If ``True`` scales the single‑qubit polynomial to lie within
        [-π, π] before applying the phase rotation.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Configured feature map circuit.

    Raises
    ------
    ValueError
        On invalid inputs (e.g., negative dimensions, unsupported basis).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be a positive integer.")
    if basis not in {"h", "ry"}:
        raise ValueError("basis must be either 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for data features
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Helper to evaluate pair weight
    def _pair_weight(i: int, j: int) -> float:
        if callable(pair_weight):
            return float(pair_weight(i, j))
        return float(pair_weight)

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:
            for q in range(n):
                qc.ry(math.pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            poly = _odd_degree_poly(x[i], single_coeffs)
            if normalise:
                # Map polynomial to [-π, π] by scaling
                poly = (2 * math.pi) * (poly / (sum(abs(c) for c in single_coeffs)))
            qc.p(2 * poly, i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions
        for (i, j) in pairs:
            weight = _pair_weight(i, j)
            angle = 2 * weight * x[i] * x[j]
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach parameter vector for external binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_poly_controlled`.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits.
    reps : int, default 2
    entanglement : str | Sequence[Tuple[int, int]] | Callable
    single_coeffs : Sequence[float], default (1.0, 0.5)
    pair_weight : float | Callable[[int, int], float], default 1.0
    basis : str, default "h"
    parameter_prefix : str, default "x"
    insert_barriers : bool, default False
    normalise : bool, default False
    name : str, default "ZZFeatureMapPolyControlled"

    Notes
    -----
    The class inherits from :class:`~qiskit.circuit.QuantumCircuit` and
    exposes the same public interface.  The parameter vector is stored as
    ``self.input_params`` for convenience.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0, 0.5),
        pair_weight: float | Callable[[int, int], float] = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalise: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            basis,
            parameter_prefix,
            insert_barriers,
            normalise,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "zz_feature_map_poly_controlled",
    "ZZFeatureMapPolyControlled",
]
