"""Extended Polynomial ZZ Feature Map with optional higher‑order interactions and scaling.

This module defines :class:`ZZFeatureMapPolyExtended` and the helper function
:func:`zz_feature_map_poly_extended`.  The implementation builds upon the
original ``zz_feature_map_poly`` but adds:

* Optional three‑qubit ZZ interactions (triplets) controlled by ``higher_order``.
* A user‑supplied scaling factor that normalises feature values.
* Flexible basis preparation (Hadamard, RY, or SX).
* Optional pre‑ and post‑rotations around the Y‑axis for each layer.
* Parameter validation and informative error messages.
* A minimal, import‑ready API compatible with Qiskit’s data‑encoding workflows.

Typical usage:
>>> from zz_feature_map_poly_extended import ZZFeatureMapPolyExtended
>>> qc = ZZFeatureMapPolyExtended(4, reps=3, higher_order=3,
...                               scaling_factor=0.5, basis='ry')
>>> qc.bind_parameters({f'x{i}': 0.1 for i in range(4)})

"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specs:
    * ``"full"`` – all‑to‑all pairs (i < j)
    * ``"linear"`` – nearest neighbors
    * ``"circular"`` – linear plus wrap‑around when ``n > 2``
    * explicit list of tuples ``[(0, 1), (2, 3)]``
    * callable ``f(n)`` returning a sequence of ``(i, j)``

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification is provided or pairs are out of range.
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


def _resolve_triplet_pairs(num_qubits: int) -> List[Tuple[int, int, int]]:
    """
    Generate all unique triplets of qubits for higher‑order interactions.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    List[Tuple[int, int, int]]
        Validated list of qubit triplets.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 3.
    """
    if num_qubits < 3:
        raise ValueError("Triplet interactions require at least 3 qubits.")
    triplets: List[Tuple[int, int, int]] = []
    for i in range(num_qubits - 2):
        for j in range(i + 1, num_qubits - 1):
            for k in range(j + 1, num_qubits):
                triplets.append((i, j, k))
    return triplets


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    higher_order: int | None = 3,
    higher_order_weight: float = 1.0,
    basis: str = "h",  # "h", "ry", or "sx"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_post_rotations: bool = False,
    scaling_factor: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with optional higher‑order interactions.

    The circuit is constructed in *reps* layers.  Each layer consists of:

    * Basis preparation (Hadamard, RY(π/2), or SX(π/2)).
    * Optional Y‑rotations before/after the main entangling block.
    * Single‑qubit phase gates with a polynomial mapping of the input features.
    * Pair‑wise ZZ rotations weighted by ``pair_weight``.
    * Optional three‑qubit ZZ rotations weighted by ``higher_order_weight``.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (and qubits).
    reps : int, default 2
        Number of repeating layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of which qubit pairs are entangled.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial mapping of single‑qubit phases.
    pair_weight : float, default 1.0
        Overall weight of pair‑wise ZZ interactions.
    higher_order : int | None, default 3
        If set to an integer >=3, enables higher‑order interactions up to that order.
        Currently only 3‑qubit interactions are supported; ``None`` disables them.
    higher_order_weight : float, default 1.0
        Weight applied to higher‑order interactions.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2), ``"sx"`` for SX(π/2).
    parameter_prefix : str, default "x"
        Prefix used for the ParameterVector.
    insert_barriers : bool, default False
        Whether to insert barriers after each logical block.
    pre_post_rotations : bool, default False
        If True, apply a Y‑rotation of π/2 before and after each layer.
    scaling_factor : float, default 1.0
        Global scaling applied to all feature values before mapping.
    name : str | None
        Optional name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding.

    Raises
    ------
    ValueError
        If input arguments are invalid (e.g., negative feature dimension, unsupported basis).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if higher_order is not None and higher_order!= 3:
        raise ValueError("Only 3‑qubit higher‑order interactions are supported; "
                         f"got {higher_order}.")
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Polynomial mapping for a single qubit
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        power: ParameterExpression = xi
        for coeff in single_coeffs:
            expr += coeff * power
            power *= xi  # next power
        return expr * scaling_factor

    # Pairwise mapping
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj * scaling_factor

    # Higher‑order mapping (currently 3‑qubit)
    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return higher_order_weight * xi * xj * xk * scaling_factor

    pairs = _resolve_entanglement(n, entanglement)
    triplets = _resolve_triplet_pairs(n) if higher_order is not None else []

    for rep in range(reps):
        # Optional pre‑rotations
        if pre_post_rotations:
            for q in range(n):
                qc.ry(pi / 2, q)

        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)
        elif basis == "sx":
            for q in range(n):
                qc.sx(q)
        else:
            raise ValueError("basis must be one of 'h', 'ry', or'sx'.")

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase gates
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ rotations
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional higher‑order ZZ rotations
        if triplets:
            for (i, j, k) in triplets:
                angle = 2 * map3(x[i], x[j], x[k])
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, j)

        # Optional post‑rotations
        if pre_post_rotations:
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    OO wrapper around :func:`zz_feature_map_poly_extended`.

    Parameters are identical to the helper function.  The instance inherits all
    methods of :class:`~qiskit.circuit.quantumcircuit.QuantumCircuit` and
    exposes ``input_params`` for easy binding.

    Example
    -------
    >>> from zz_feature_map_poly_extended import ZZFeatureMapPolyExtended
    >>> qc = ZZFeatureMapPolyExtended(4, reps=2, higher_order=3)
    >>> qc.bind_parameters({f'x{i}': 0.1 for i in range(4)})
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        higher_order: int | None = 3,
        higher_order_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_post_rotations: bool = False,
        scaling_factor: float = 1.0,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension, reps, entanglement, single_coeffs, pair_weight,
            higher_order, higher_order_weight, basis, parameter_prefix,
            insert_barriers, pre_post_rotations, scaling_factor, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
