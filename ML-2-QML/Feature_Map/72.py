"""Extended Polynomial ZZ Feature Map with optional higher‑order interactions and pre/post rotations."""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
import itertools


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``\"full\"``: all‑to‑all pairs (i < j)
      - ``\"linear\"``: nearest neighbours (0,1), (1,2),...
      - ``\"circular\"``: linear plus wrap‑around (n‑1,0) if n > 2
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
        return [(int(i), int(j)) for (i, j) in entanglement(num_qubits)]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = x · y."""
    return x * y


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = x · y · z."""
    return x * y * z


# ---------------------------------------------------------------------------
# Extended feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_ext(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    three_qubit_weight: float = 0.0,
    include_three_qubit: bool = False,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotation: bool = False,
    pre_rotation_angle: float = pi / 4,
    post_rotation: bool = False,
    post_rotation_angle: float = pi / 4,
    normalize: bool = False,
    normalisation_factor: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be ≥ 2.
    reps : int, default 2
        Number of repeated feature‑map layers.
    entanglement : str | sequence | callable, default "full"
        Specification of two‑qubit entanglement pattern.
    single_coeffs : sequence of float, default (1.0,)
        Coefficients for the polynomial φ1(x) = Σ c_k · x^{k+1}.
    pair_weight : float, default 1.0
        Weight for the two‑qubit ZZ interaction φ2(x, y) = pair_weight · x · y.
    three_qubit_weight : float, default 0.0
        Weight for optional three‑qubit ZZ interaction φ3(x, y, z) = three_qubit_weight · x · y · z.
    include_three_qubit : bool, default False
        Whether to include the three‑qubit interactions.
    basis : str, default "h"
        Basis preparation: ``\"h\"`` for Hadamard or ``\"ry\"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        Insert barriers between logical layers for visual clarity.
    pre_rotation : bool, default False
        Apply a global RZ rotation before the first layer.
    pre_rotation_angle : float, default π/4
        Angle for the pre‑rotation RZ gate.
    post_rotation : bool, default False
        Apply a global RZ rotation after the last layer.
    post_rotation_angle : float, default π/4
        Angle for the post‑rotation RZ gate.
    normalize : bool, default False
        If True, divide all feature values by ``normalisation_factor`` before encoding.
    normalisation_factor : float, default 1.0
        Scaling factor applied when ``normalize=True``.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    * The circuit exposes ``input_params`` for parameter binding.
    * Raises ValueError if ``feature_dimension`` < 2 when three‑qubit interactions are requested.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if include_three_qubit and feature_dimension < 3:
        raise ValueError("Three‑qubit interactions require feature_dimension >= 3.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExt")

    x = ParameterVector(parameter_prefix, n)

    # Optional normalisation
    scale = 1.0 / normalisation_factor if normalize else 1.0

    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi  # next power
        return expr * scale

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj * scale

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return three_qubit_weight * xi * xj * xk * scale

    # Two‑qubit entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Three‑qubit combination list
    three_qubit_pairs: List[Tuple[int, int, int]] = []
    if include_three_qubit:
        three_qubit_pairs = list(itertools.combinations(range(n), 3))

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if rep == 0 and pre_rotation:
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

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional three‑qubit ZZ via CCX–P–CCX
        if include_three_qubit:
            for (i, j, k) in three_qubit_pairs:
                angle = 2 * map3(x[i], x[j], x[k])
                qc.ccx(i, j, k)
                qc.p(angle, k)
                qc.ccx(i, j, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Optional post‑rotation
    if post_rotation:
        for q in range(n):
            qc.rz(post_rotation_angle, q)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Object‑oriented wrapper for :func:`zz_feature_map_poly_ext`.

    Parameters
    ----------
    All parameters are forwarded to :func:`zz_feature_map_poly_ext` and are documented there.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        three_qubit_weight: float = 0.0,
        include_three_qubit: bool = False,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotation: bool = False,
        pre_rotation_angle: float = pi / 4,
        post_rotation: bool = False,
        post_rotation_angle: float = pi / 4,
        normalize: bool = False,
        normalisation_factor: float = 1.0,
        name: str = "ZZFeatureMapPolyExt",
    ) -> None:
        built = zz_feature_map_poly_ext(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            three_qubit_weight,
            include_three_qubit,
            basis,
            parameter_prefix,
            insert_barriers,
            pre_rotation,
            pre_rotation_angle,
            post_rotation,
            post_rotation_angle,
            normalize,
            normalisation_factor,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_ext"]
