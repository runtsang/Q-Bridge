"""ZZFeatureMapPolyControlled – a controlled‑pair, symmetric polynomial feature map.

This module provides a feature map that enhances the baseline polynomial
ZZFeatureMap by:
  * Using a two‑stage polynomial for each qubit (pre‑ and post‑rotation).
  * Applying a symmetric pairwise interaction that is shared between the two qubits
    via a single angle θ_{ij} = pair_weight * x_i * x_j.
  * Introducing an optional threshold that zeroes out pairwise angles
    when |θ_{ij}| < pair_threshold, effectively controlling the sparsity of
    entanglement.
  * Maintaining the same circuit depth and entanglement pattern as the original,
    but with the added ability to gate interactions.

The module exposes a functional interface `zz_feature_map_poly_controlled` and a
class `ZZFeatureMapPolyControlled` that inherits from QuantumCircuit.
Both interfaces support parameter binding via the `input_params` attribute.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import sympy
from sympy import Abs, Piecewise

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


def _pair_angle(
    xi: ParameterExpression,
    xj: ParameterExpression,
    pair_weight: float,
    pair_threshold: float | None,
) -> ParameterExpression:
    """Compute the pairwise interaction angle with optional thresholding.

    If ``pair_threshold`` is not None, the returned expression is
    ``Piecewise((0, Abs(raw) < pair_threshold), (raw, True))`` where
    ``raw = pair_weight * xi * xj``.  This effectively zeros out
    interactions whose magnitude falls below the threshold.
    """
    raw = pair_weight * xi * xj
    if pair_threshold is not None:
        return Piecewise((0, Abs(raw) < pair_threshold), (raw, True))
    return raw


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_threshold: float | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a polynomial ZZ feature map with controlled pairwise interactions.

    Parameters
    ----------
    feature_dimension
        Number of qubits / dimensionality of the input vector.
    reps
        Number of repetitions of the feature map block.
    entanglement
        Specification of qubit pairs to entangle.
    single_coeffs
        Coefficients for the polynomial φ₁(x) = Σ a_k x^{k+1}.
    pair_weight
        Global scaling factor for the pairwise interaction.
    basis
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix
        Prefix for the ParameterVector that holds the classical features.
    insert_barriers
        Whether to insert barriers between layers.
    pair_threshold
        If set, pairwise angles whose magnitude falls below this value are
        replaced with zero, effectively gating the interaction.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature map circuit.  The circuit has an attribute
        ``input_params`` containing the ParameterVector for binding.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Helper for the single‑qubit polynomial
    def poly1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        power: ParameterExpression = xi
        for coeff in single_coeffs:
            expr += coeff * power
            power *= xi
        return expr

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

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

        # Single‑qubit phase rotations
        for i in range(n):
            angle = 2 * poly1(x[i])
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * _pair_angle(x[i], x[j], pair_weight, pair_threshold)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# OO interface
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Object‑oriented wrapper for the controlled polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input vector.
    reps : int, optional
        Number of repetitions of the feature map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Specification of qubit pairs to entangle.
    single_coeffs : Sequence[float], optional
        Coefficients for the polynomial φ₁(x).
    pair_weight : float, optional
        Global scaling factor for the pairwise interaction.
    basis : str, optional
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, optional
        Prefix for the ParameterVector that holds the classical features.
    insert_barriers : bool, optional
        Whether to insert barriers between layers.
    pair_threshold : float | None, optional
        Threshold for gating pairwise interactions.
    name : str, optional
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
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_threshold: float | None = None,
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
            pair_threshold,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
