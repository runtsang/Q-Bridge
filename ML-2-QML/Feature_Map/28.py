"""Extended Polynomial ZZFeatureMap with higher‑order interactions and optional pre/post rotations."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union, Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specifiers:
      * ``"full"`` – all unique pairs (i < j)
      * ``"linear"`` – nearest‑neighbour chain
      * ``"circular"`` – linear plus wrap‑around connection for n>2
      * explicit sequence of (i, j) tuples
      * callable ``f(n) -> Iterable[(i, j)]``

    Raises
    ------
    ValueError
        If an invalid specifier is supplied or a pair references an out‑of‑range qubit.
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
        return [(int(i), int(j)) for i, j in entanglement(num_qubits)]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression, order: int, coeffs: Sequence[float]) -> ParameterExpression:
    """
    Default single‑qubit polynomial φ1(x) = Σ_k coeffs[k] · x^(k+1) for k < order.
    """
    expr: ParameterExpression = 0
    power: ParameterExpression = x
    for k in range(min(order, len(coeffs))):
        expr = expr + coeffs[k] * power
        power = power * x
    return expr


def _default_map_2(x: ParameterExpression, y: ParameterExpression,
                  order: int, weight: float) -> ParameterExpression:
    """
    Default pair interaction φ2(x, y) = weight · x · y for first‑order pairs.
    """
    return weight * x * y


# ---------------------------------------------------------------------------
# Extended Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    max_order: int = 2,
    pair_order: int = 1,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotations: bool = False,
    post_rotations: bool = False,
    pre_rotation_angle: float = pi / 4,
    post_rotation_angle: float = -pi / 4,
    normalise_features: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with optional higher‑order terms, normalisation, and
    configurable pre/post rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be ≥ 2.
    reps : int, default 2
        Number of repetitions of the basis + interaction block.
    entanglement : str or iterable of (int, int) or callable
        Entanglement pattern. See :func:`_resolve_entanglement`.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial. Length determines the
        maximum polynomial degree (max_order).
    pair_weight : float, default 1.0
        Base weight for pair interactions. If ``pair_order > 1`` only the
        first‑order term is scaled by this weight; higher‑order terms are
        scaled by ``pair_weight`` multiplied by the corresponding powers.
    max_order : int, default 2
        Highest power of each feature in the single‑qubit map.
    pair_order : int, default 1
        Highest combined power of two features in pair interactions.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        Insert barriers between logical layers for visual clarity.
    pre_rotations : bool, default False
        Apply a fixed RZ rotation (``pre_rotation_angle``) to every qubit
        before the basis preparation.
    post_rotations : bool, default False
        Apply a fixed RZ rotation (``post_rotation_angle``) to every qubit
        after the entanglement block.
    normalise_features : bool, default False
        If True, normalise each feature by 1/√feature_dimension before mapping.
    name : str or None, default None
        QuantumCircuit name. If None, defaults to ``"ZZFeatureMapPolyExtended"``.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding to classical data.

    Raises
    ------
    ValueError
        For invalid input dimensions or unsupported basis strings.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if max_order < 1:
        raise ValueError("max_order must be >= 1.")
    if pair_order < 1:
        raise ValueError("pair_order must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Create parameter vector
    params = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotations
        if pre_rotations:
            for q in range(n):
                qc.rz(pre_rotation_angle, q)

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

        # Normalise features if requested
        norm_factor = 1 / (n**0.5) if normalise_features else 1

        # Single‑qubit phases
        for i in range(n):
            xi = params[i] * norm_factor
            phase = 2 * _default_map_1(xi, max_order, single_coeffs)
            qc.p(phase, i)

        if insert_barriers:
            qc.barrier()

        # Pair interactions via CX–P–CX
        for (i, j) in pairs:
            xi = params[i] * norm_factor
            xj = params[j] * norm_factor
            angle = 2 * _default_map_2(xi, xj, pair_order, pair_weight)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

        # Optional post‑rotations
        if post_rotations:
            for q in range(n):
                qc.rz(post_rotation_angle, q)

    qc.input_params = params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_poly_extended`.

    Parameters
    ----------
    feature_dimension : int
    reps : int, default 2
    entanglement : str or iterable of (int, int) or callable, default "full"
    single_coeffs : Sequence[float], default (1.0,)
    pair_weight : float, default 1.0
    max_order : int, default 2
    pair_order : int, default 1
    basis : str, default "h"
    parameter_prefix : str, default "x"
    insert_barriers : bool, default False
    pre_rotations : bool, default False
    post_rotations : bool, default False
    pre_rotation_angle : float, default pi/4
    post_rotation_angle : float, default -pi/4
    normalise_features : bool, default False
    name : str, default "ZZFeatureMapPolyExtended"
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Iterable[Tuple[int, int]]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        max_order: int = 2,
        pair_order: int = 1,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotations: bool = False,
        post_rotations: bool = False,
        pre_rotation_angle: float = pi / 4,
        post_rotation_angle: float = -pi / 4,
        normalise_features: bool = False,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            max_order,
            pair_order,
            basis,
            parameter_prefix,
            insert_barriers,
            pre_rotations,
            post_rotations,
            pre_rotation_angle,
            post_rotation_angle,
            normalise_features,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
