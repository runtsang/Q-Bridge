"""
ZZFeatureMapPolyControlled: Controlled‑modified polynomial ZZ feature map for Qiskit.

This module provides a feature‑map circuit that extends the original polynomial
ZZ feature map by adding optional symmetrised entanglement, pre‑rotations,
data scaling, and a global phase factor.  It is fully compatible with
Qiskit’s data‑encoding workflow and exposes both a functional helper
(`zz_feature_map_poly_controlled`) and an OO wrapper (`ZZFeatureMapPolyControlled`).
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
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2),...
      - "circular": linear plus wrap‑around (n-1,0) if n > 2
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
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _symmetrised_pairs(num_qubits: int) -> List[Tuple[int, int]]:
    """Generate symmetrised entanglement pairs (i, n-1-i)."""
    return [(i, num_qubits - 1 - i) for i in range(num_qubits // 2)]


# ---------------------------------------------------------------------------
# Feature‑Map Construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    symmetrised_entanglement: bool = False,
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    data_scale: float = 1.0,
    pre_rotations: Sequence[ParameterExpression] | None = None,
    rotation_basis: str = "ry",
    basis: str = "h",
    insert_barriers: bool = False,
    name: str | None = None,
    parameter_prefix: str = "x",
) -> QuantumCircuit:
    """Return a polynomial ZZ feature‑map with optional symmetrised entanglement,
    pre‑rotations, and data scaling.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification. Ignored when ``symmetrised_entanglement`` is True.
    symmetrised_entanglement : bool, default False
        If True, use only symmetric qubit pairs (i, n-1-i) for entanglement.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial mapping of single‑qubit phases.
    pair_weight : float, default 1.0
        Weight applied to each pair interaction.  All pairs share the same weight.
    data_scale : float, default 1.0
        Global scaling factor applied to the input data before mapping.
    pre_rotations : Sequence[ParameterExpression] | None, default None
        Optional pre‑rotation angles applied to each qubit before the main encoding.
        Length must match ``feature_dimension``.
    rotation_basis : str, default "ry"
        Basis for the pre‑rotations: ``"ry"`` or ``"rx"``.
    basis : str, default "h"
        Basis preparation for the main encoding: ``"h"`` (Hadamard) or ``"ry"``.
    insert_barriers : bool, default False
        If True, insert ``Barrier`` instructions between logical blocks.
    name : str | None, default None
        Name of the resulting circuit.  If None, a default name is used.
    parameter_prefix : str, default "x"
        Prefix for the automatically generated parameter vector.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit ready for binding with classical data.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pre_rotations is not None and len(pre_rotations)!= n:
        raise ValueError("pre_rotations length must match number of qubits.")
    if rotation_basis not in ("ry", "rx"):
        raise ValueError("rotation_basis must be 'ry' or 'rx'.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")
    x = ParameterVector(parameter_prefix, n)

    def map1(xi: ParameterExpression) -> ParameterExpression:
        # Polynomial mapping for single‑qubit phase
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        # Pair interaction mapping
        return pair_weight * xi * xj

    if symmetrised_entanglement:
        pairs = _symmetrised_pairs(n)
    else:
        pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotations
        if pre_rotations is not None:
            for i, angle in enumerate(pre_rotations):
                if rotation_basis == "ry":
                    qc.ry(angle, i)
                else:  # rotation_basis == "rx"
                    qc.rx(angle, i)

        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # basis == "ry"
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]) * data_scale, i)

        if insert_barriers:
            qc.barrier()

        # ZZ pair interactions
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j]) * data_scale**2
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified polynomial ZZ feature map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        symmetrised_entanglement: bool = False,
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        data_scale: float = 1.0,
        pre_rotations: Sequence[ParameterExpression] | None = None,
        rotation_basis: str = "ry",
        basis: str = "h",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
        parameter_prefix: str = "x",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            symmetrised_entanglement,
            single_coeffs,
            pair_weight,
            data_scale,
            pre_rotations,
            rotation_basis,
            basis,
            insert_barriers,
            None,  # name is handled by super()
            parameter_prefix,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
