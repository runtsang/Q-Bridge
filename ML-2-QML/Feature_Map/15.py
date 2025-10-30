"""Extended Polynomial ZZ Feature Map with Higher‑Order Interactions.

This module provides a Qiskit‑compatible feature map that extends the
`ZZFeatureMapPoly` seed by:
  • Polynomial single‑qubit terms up to an arbitrary degree.
  • Pairwise interactions up to `pair_degree` with custom coefficients.
  • Optional three‑body interactions up to `triple_degree`.
  • Flexible entanglement patterns (full, linear, circular, explicit, or
    callable).
  • Pre‑ and post‑ rotations in the Hadamard (RY) basis or identity.
  • Automatic depth selection when `reps="auto"`.
  • Normalization toggle to keep phase angles bounded.
  • Barrier insertion for visual layer separation.

The implementation is fully Qiskit‑compatible: the returned circuit exposes
`input_params` for parameter binding, and a class wrapper `ZZFeatureMapPolyExtended`
is provided for object‑oriented use.

"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

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
      - explicit list of pairs like ``[(0, 2), (1, 3)]``
      - callable: ``f(num_qubits) -> sequence of (i, j)``
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


def _default_map1(x: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
    """Polynomial φ1(x) = Σ_k coeffs[k] * x^{k+1}."""
    expr: ParameterExpression = 0
    power = x
    for c in coeffs:
        expr += c * power
        power *= x
    return expr


def _default_map2(
    xi: ParameterExpression,
    xj: ParameterExpression,
    coeffs: Sequence[float],
) -> ParameterExpression:
    """Pairwise φ2(xi, xj) = Σ_k coeffs[k] * (xi * xj)^k."""
    expr: ParameterExpression = 0
    prod = xi * xj
    for c in coeffs:
        expr += c * prod
        prod *= xi * xj
    return expr


def _default_map3(
    xi: ParameterExpression,
    xj: ParameterExpression,
    xl: ParameterExpression,
    coeffs: Sequence[float],
) -> ParameterExpression:
    """Three‑body φ3(xi, xj, xl) = Σ_k coeffs[k] * (xi * xj * xl)^k."""
    expr: ParameterExpression = 0
    prod = xi * xj * xl
    for c in coeffs:
        expr += c * prod
        prod *= xi * xj * xl
    return expr


# ---------------------------------------------------------------------------
# Extended Polynomial ZZ Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int | str = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_degree: int = 1,
    pair_coeffs: Sequence[float] | None = None,
    triple_degree: int = 0,
    triple_coeffs: Sequence[float] | None = None,
    basis: str = "h",  # "h" or "ry"
    pre_rotations: str = "none",  # "none", "rz", "ry"
    post_rotations: str = "none",  # "none", "rz", "ry"
    normalize: bool = True,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map with optional higher‑order terms.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int | str, default 2
        Number of repeated layers.  ``"auto"`` selects ``ceil(n/2)``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit coupling pairs.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial φ1.
    pair_degree : int, default 1
        Highest power for pairwise interactions.
    pair_coeffs : Sequence[float] | None, default None
        Coefficients for the pairwise polynomial φ2.  Length must equal
        ``pair_degree``; otherwise defaults to all ones.
    triple_degree : int, default 0
        Highest power for three‑body interactions.
    triple_coeffs : Sequence[float] | None, default None
        Coefficients for the triple polynomial φ3.  Length must equal
        ``triple_degree``; otherwise defaults to all ones.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    pre_rotations : str, default "none"
        Pre‑layer rotation applied to each qubit: ``"none"``, ``"rz"``, ``"ry"``.
    post_rotations : str, default "none"
        Post‑layer rotation applied to each qubit: ``"none"``, ``"rz"``, ``"ry"``.
    normalize : bool, default True
        If True, scale all phase angles by ``1/π`` to keep them bounded.
    insert_barriers : bool, default False
        Insert barriers between logical layers for readability.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        A Qiskit circuit ready for parameter binding.

    Notes
    -----
    * The circuit exposes ``input_params`` for binding classical feature
      vectors.  Example:
        >>> qc = zz_feature_map_poly_extended(3)
        >>> qc.bind_parameters({qc.input_params[i]: val for i, val in enumerate([0.1, 0.2, 0.3])})
    * The normalisation factor is applied uniformly to all map outputs.
    * The circuit depth grows with ``reps`` and the degree of pair/triple terms.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    if isinstance(reps, str):
        if reps.lower() == "auto":
            reps = max(1, (n + 1) // 2)
        else:
            raise ValueError(f"Unsupported reps string: {reps!r}")

    if reps < 1:
        raise ValueError("reps must be >= 1.")

    # Validate coefficient lengths
    if pair_degree < 1:
        raise ValueError("pair_degree must be >= 1.")
    if pair_coeffs is None:
        pair_coeffs = (1.0,) * pair_degree
    if len(pair_coeffs)!= pair_degree:
        raise ValueError("Length of pair_coeffs must equal pair_degree.")

    if triple_degree < 0:
        raise ValueError("triple_degree must be >= 0.")
    if triple_degree > 0:
        if triple_coeffs is None:
            triple_coeffs = (1.0,) * triple_degree
        if len(triple_coeffs)!= triple_degree:
            raise ValueError("Length of triple_coeffs must equal triple_degree.")
    else:
        triple_coeffs = ()

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")
    x = ParameterVector("x", n)

    # Pre‑rotations
    if pre_rotations.lower() == "rz":
        for q in range(n):
            qc.rz(pi / 2, q)
    elif pre_rotations.lower() == "ry":
        for q in range(n):
            qc.ry(pi / 2, q)
    elif pre_rotations.lower()!= "none":
        raise ValueError("pre_rotations must be one of 'none', 'rz', 'ry'.")

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)
        else:
            raise ValueError("basis must be 'h' or 'ry'.")

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            angle = 2 * _default_map1(x[i], single_coeffs)
            if normalize:
                angle /= pi
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ interactions
        for (i, j) in pairs:
            angle = 2 * _default_map2(x[i], x[j], pair_coeffs)
            if normalize:
                angle /= pi
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Three‑body interactions (if any)
        if triple_degree > 0:
            indices = list(range(n))
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        angle = 2 * _default_map3(x[i], x[j], x[k], triple_coeffs)
                        if normalize:
                            angle /= pi
                        # Implement a simple 3‑body ZZ via two CNOTs and a phase
                        qc.cx(i, j)
                        qc.cx(j, k)
                        qc.p(angle, k)
                        qc.cx(j, k)
                        qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Post‑rotations
    if post_rotations.lower() == "rz":
        for q in range(n):
            qc.rz(pi / 2, q)
    elif post_rotations.lower() == "ry":
        for q in range(n):
            qc.ry(pi / 2, q)
    elif post_rotations.lower()!= "none":
        raise ValueError("post_rotations must be one of 'none', 'rz', 'ry'.")

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Object‑oriented wrapper for the extended polynomial ZZ feature map.

    Parameters
    ----------
    All arguments are passed directly to :func:`zz_feature_map_poly_extended`.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int | str = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_degree: int = 1,
        pair_coeffs: Sequence[float] | None = None,
        triple_degree: int = 0,
        triple_coeffs: Sequence[float] | None = None,
        basis: str = "h",
        pre_rotations: str = "none",
        post_rotations: str = "none",
        normalize: bool = True,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_degree,
            pair_coeffs,
            triple_degree,
            triple_coeffs,
            basis,
            pre_rotations,
            post_rotations,
            normalize,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
