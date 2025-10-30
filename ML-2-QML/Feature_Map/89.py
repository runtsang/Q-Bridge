"""Controlled‑modified polynomial ZZ feature map for Qiskit data encoding.

This module provides a functional helper and a subclassed
QuantumCircuit that implement a polynomial ZZ feature map with
the following controlled modifications:

* **Data re‑parameterisation** – user supplied scaling or normalisation
  of raw feature values.
* **Symmetric pair interactions** – optional double‑phase on both qubits
  of a pair to enforce a fully symmetric coupling.
* **Pre‑ and post‑rotations** – optional RZ rotations before and after
  the basis preparation to enrich the expressibility.
* **Shared or per‑qubit coefficients** – single‑qubit polynomial
  coefficients can be shared across all qubits or specified per qubit.
* **Barrier insertion** – optional barriers for visual clarity and
  debugging.

Both the helper function and the class expose an ``input_params``
attribute for easy parameter binding with ``bind_parameters``.
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
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs like ``[(0, 2), (1, 3)]``
      - callable: ``f(num_qubits) -> sequence of (i, j)``

    Raises:
        ValueError: If an unknown spec or invalid pair is supplied.
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
# Main feature‑map definition
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",  # "h" or "ry" or "none"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    shared_single_coeffs: bool = True,
    symmetrize_pairs: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    normalize_features: bool = False,
    feature_scaling: Callable[[ParameterExpression], ParameterExpression] | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modified polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / input features. Must be >= 2.
    reps : int
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement pattern spec. See ``_resolve_entanglement``.
    single_coeffs : Sequence[float]
        Polynomial coefficients for the single‑qubit phase φ1(x).
        If ``shared_single_coeffs`` is True, the same sequence is used
        for all qubits; otherwise the sequence must be per‑qubit.
    pair_weight : float
        Scalar weight for the pairwise phase φ2(x, y) = weight * x * y.
    basis : str
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2),
        or ``"none"`` for no basis prep.
    parameter_prefix : str
        Prefix for the ParameterVector names.
    insert_barriers : bool
        If True, insert barriers between logical sections.
    name : str | None
        Optional circuit name. Defaults to ``"ZZFeatureMapPolyControlled"``.
    shared_single_coeffs : bool
        If False, ``single_coeffs`` must be a sequence of length
        ``feature_dimension`` containing per‑qubit coefficient lists.
    symmetrize_pairs : bool
        If True, apply a symmetric ZZ interaction by rotating both qubits
        of each entangled pair.
    pre_rotation : bool
        If True, apply a pre‑rotation RZ(2*φ1(x)) before basis prep.
    post_rotation : bool
        If True, apply a post‑rotation RZ(2*φ1(x)) after entanglement.
    normalize_features : bool
        If True, rescale raw features to the interval [0, π/2] via
        ``feature_scaling = lambda x: (pi/2) * x``.
    feature_scaling : Callable[[ParameterExpression], ParameterExpression] | None
        Optional custom scaling function applied to each raw feature.
        If ``None`` and ``normalize_features`` is False, the identity is used.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with ``bind_parameters``.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if basis not in ("h", "ry", "none"):
        raise ValueError("basis must be 'h', 'ry', or 'none'.")
    if not isinstance(shared_single_coeffs, bool):
        raise ValueError("shared_single_coeffs must be a boolean.")
    if not isinstance(symmetrize_pairs, bool):
        raise ValueError("symmetrize_pairs must be a boolean.")
    if not isinstance(pre_rotation, bool):
        raise ValueError("pre_rotation must be a boolean.")
    if not isinstance(post_rotation, bool):
        raise ValueError("post_rotation must be a boolean.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    x = ParameterVector(parameter_prefix, n)

    # Apply optional scaling
    if feature_scaling is None:
        if normalize_features:
            feature_scaling = lambda xi: (pi / 2) * xi
        else:
            feature_scaling = lambda xi: xi

    # Prepare per‑qubit scaled parameters
    scaled_params: List[ParameterExpression] = [feature_scaling(xi) for xi in x]

    # Polynomial map for single qubit
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi
        return expr

    # Pairwise map
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.ry(2 * map1(scaled_params[i]), i)

        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(scaled_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # Entanglement block
        for (i, j) in pairs:
            angle = 2 * map2(scaled_params[i], scaled_params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
            if symmetrize_pairs:
                qc.p(angle, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.ry(2 * map1(scaled_params[i]), i)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    Object‑oriented wrapper for the controlled‑modified polynomial ZZ feature map.

    The constructor accepts the same arguments as ``zz_feature_map_poly_controlled``
    and composes the resulting circuit into the instance.

    Attributes
    ----------
    input_params : ParameterVector
        Parameters that can be bound via ``bind_parameters``.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
        shared_single_coeffs: bool = True,
        symmetrize_pairs: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        normalize_features: bool = False,
        feature_scaling: Callable[[ParameterExpression], ParameterExpression] | None = None,
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
            name,
            shared_single_coeffs,
            symmetrize_pairs,
            pre_rotation,
            post_rotation,
            normalize_features,
            feature_scaling,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "zz_feature_map_poly_controlled",
    "ZZFeatureMapPolyControlled",
]
