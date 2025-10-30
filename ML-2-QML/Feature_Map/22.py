"""Quantum feature map: ZZFeatureMapPolyExtended

An extended polynomial ZZ feature map that supports up to third‑order
interactions, optional feature scaling, and configurable entanglement.
The implementation follows Qiskit’s data‑encoding conventions and
provides both a functional helper and a class‑based wrapper.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``    : all‑to‑all pairs (i < j)
      - ``"linear"``  : nearest neighbors (0,1), (1,2), …
      - ``"circular"``: linear + wrap‑around (n-1,0) if n > 2
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


def _resolve_triplets(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Return all unique triples (i, j, k) with i < j < k."""
    return [(i, j, k) for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            for k in range(j + 1, num_qubits)]


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 0.0,
    interaction_order: int = 2,
    feature_scale: float = 1.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with optional higher‑order interactions.

    The map is parameterised by a feature vector ``x`` of length
    ``feature_dimension``.  The circuit is built from the following
    building blocks:

    * **Basis preparation** – Hadamard or RY(π/2) on all qubits.
    * **Single‑qubit phases** – φ₁(xᵢ) = Σₖ cₖ·(xᵢ·feature_scale)^(k+1)
    * **Pairwise ZZ** – φ₂(xᵢ, xⱼ) = pair_weight·(xᵢ·xⱼ)·feature_scale²
    * **Triplet ZZ** – φ₃(xᵢ, xⱼ, xₖ) = triple_weight·(xᵢ·xⱼ·xₖ)·feature_scale³
      (enabled when ``interaction_order`` ≥ 3)

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be ≥ 2).
    reps : int, default 2
        Number of feature‑map repetitions (depth).
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Defines which qubit pairs receive ZZ interactions.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial mapping φ₁.
    pair_weight : float, default 1.0
        Overall weight for pairwise ZZ terms.
    triple_weight : float, default 0.0
        Weight for third‑order ZZ terms (ignored if ``interaction_order`` < 3).
    interaction_order : int, default 2
        Allowed values: 1 (single‑qubit only), 2 (pairwise), 3 (triplet).
    feature_scale : float, default 1.0
        Global scaling factor applied to all feature inputs.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter names in the :class:`ParameterVector`.
    insert_barriers : bool, default False
        Whether to insert barriers between circuit sections.
    name : str | None, default None
        Optional name for the constructed circuit.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data binding.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``reps`` ≤ 0, or
        ``interaction_order`` not in {1, 2, 3}.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    if interaction_order not in (1, 2, 3):
        raise ValueError("interaction_order must be 1, 2, or 3.")
    if interaction_order == 3 and triple_weight == 0.0:
        raise ValueError("triple_weight must be non‑zero when interaction_order==3.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")
    x = ParameterVector(parameter_prefix, n)

    # Helper mappings
    def map1(xi: ParameterExpression) -> ParameterExpression:
        """φ₁(xᵢ) = Σₖ cₖ·(xi·scale)^(k+1)"""
        expr: ParameterExpression = 0
        p = xi * feature_scale
        for c in single_coeffs:
            expr = expr + c * p
            p = p * (xi * feature_scale)  # next power
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """φ₂(xᵢ, xⱼ) = pair_weight·xi·xj·scale²"""
        return pair_weight * xi * xj * feature_scale * feature_scale

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        """φ₃(xᵢ, xⱼ, xₖ) = triple_weight·xi·xj·xk·scale³"""
        return triple_weight * xi * xj * xk * feature_scale ** 3

    pairs = _resolve_entanglement(n, entanglement)
    triples = _resolve_triplets(n) if interaction_order == 3 else []

    for rep in range(int(reps)):
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
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ (CX–P–CX)
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional third‑order ZZ (CCX–P–CCX)
        if interaction_order == 3:
            for (i, j, k) in triples:
                angle = 2 * map3(x[i], x[j], x[k])
                qc.ccx(i, j, k)
                qc.p(angle, k)
                qc.ccx(i, j, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑based wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_poly_extended`.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be ≥ 2).
    reps : int, default 2
        Number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement specification.
    single_coeffs : Sequence[float], default (1.0,)
        Polynomial coefficients for φ₁.
    pair_weight : float, default 1.0
        Weight for pairwise ZZ terms.
    triple_weight : float, default 0.0
        Weight for third‑order ZZ terms.
    interaction_order : int, default 2
        Interaction order {1, 2, 3}.
    feature_scale : float, default 1.0
        Global scaling factor for input features.
    basis : str, default "h"
        Basis preparation: ``"h"`` or ``"ry"``.
    parameter_prefix : str, default "x"
        Prefix for the parameter names.
    insert_barriers : bool, default False
        Whether to insert barriers.
    name : str, default "ZZFeatureMapPolyExtended"
        Circuit name.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 0.0,
        interaction_order: int = 2,
        feature_scale: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triple_weight,
            interaction_order,
            feature_scale,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
