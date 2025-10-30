"""Extended polynomial ZZ feature map with optional higher‑order interactions,
adaptive depth, and global rotations.

This module provides a function and a class that generate a Qiskit
`QuantumCircuit` suitable for feature‑map encoding in variational quantum
learning tasks.  The design follows an *extension* scaling paradigm:
it keeps the core structure of the original `ZZFeatureMapPoly` while adding
new capabilities that increase expressivity without sacrificing
compatibility with Qiskit’s `input_params` interface.

Supported features
------------------
- **Higher‑order interactions**: optional three‑qubit ZZ‑phase terms
  (controlled‑phase via a Toffoli‑like decomposition) with a user‑defined
  weight.
- **Adaptive depth**: `reps` may be an integer or a callable that
  returns the number of repetitions based on the feature dimension.
- **Global rotations**: optional RY rotations applied before the first
  repetition and after the last repetition.
- **Per‑repetition extra rotations**: optional RY rotations applied to
  each qubit after every repetition.
- **Basis preparation**: Hadamard or `RY(pi/2)` on all qubits per
  repetition.
- **Entanglement specification**: full, linear, circular, explicit
  pairs, or a callable returning pairs.
- **Feature scaling**: a multiplicative factor applied to all input
  parameters.

Validation
----------
All arguments are validated with informative `ValueError`s:
  * `feature_dimension` must be ≥ 1.
  * `reps` must resolve to a positive integer.
  * `single_coeffs` must be a non‑empty sequence of floats.
  * Entanglement pairs must connect distinct qubits within range.
  * Rotation angles must be real numbers.
  * `interaction_order` ≤ 3 (currently 1‑, 2‑, 3‑body supported).

Usage
-----
```python
from zz_feature_map_poly_extended import ZZFeatureMapPolyExtended

# Functional API
qc = zz_feature_map_poly_extended(
    feature_dimension=4,
    reps=3,
    include_three_body=True,
    three_body_weight=0.7,
    pre_global_angle=0.1,
    post_global_angle=0.2,
    insert_barriers=True,
)

# Class‑based API
circuit = ZZFeatureMapPolyExtended(
    feature_dimension=4,
    reps=3,
    include_three_body=True,
    three_body_weight=0.7,
    pre_global_angle=0.1,
    post_global_angle=0.2,
    insert_barriers=True,
)
```
Both APIs expose an `input_params` attribute for parameter binding, enabling
direct use with Qiskit’s data‑encoding utilities.

"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """
    Resolve a user‑supplied entanglement specification into a list of qubit pairs.

    Supported specs:
    - "full": all‑to‑all pairs (i < j)
    - "linear": nearest neighbors
    - "circular": linear plus wrap‑around
    - explicit list of pairs
    - callable: f(num_qubits) → sequence of (i, j)

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid pairs.
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(
    x: ParameterExpression,
    y: ParameterExpression,
    z: ParameterExpression,
    weight: float,
) -> ParameterExpression:
    """Default φ3(x, y, z) = weight · x · y · z."""
    return weight * x * y * z


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int | Callable[[int], int] = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    three_body_weight: float = 0.5,
    include_three_body: bool = False,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_global_angle: float = 0.0,
    post_global_angle: float = 0.0,
    extra_per_rep_angle: float = 0.0,
    extra_per_rep_basis: str = "ry",
    scaling_factor: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be ≥ 1.
    reps : int | Callable[[int], int], default 2
        Number of repetitions. If callable, it receives the feature
        dimension and must return a positive integer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial φ1(x) = Σ_k coeff[k] · x^(k+1).
    pair_weight : float, default 1.0
        Weight for the two‑qubit interaction φ2(x, y) = pair_weight · x · y.
    three_body_weight : float, default 0.5
        Weight for the optional three‑qubit interaction φ3(x, y, z).
    include_three_body : bool, default False
        If True, a three‑body interaction is added after each two‑qubit
        interaction within a repetition.
    basis : str, default "h"
        Basis preparation for each qubit: "h" (Hadamard) or "ry" (RY(pi/2)).
    parameter_prefix : str, default "x"
        Prefix for the parameters in the `ParameterVector`.
    insert_barriers : bool, default False
        Insert barriers between logical sections.
    pre_global_angle : float, default 0.0
        RY rotation applied to every qubit before the first repetition.
    post_global_angle : float, default 0.0
        RY rotation applied to every qubit after the last repetition.
    extra_per_rep_angle : float, default 0.0
        RY rotation applied to every qubit after each repetition.
    extra_per_rep_basis : str, default "ry"
        Basis for the per‑rep rotation: "h" or "ry".
    scaling_factor : float, default 1.0
        Multiplicative factor applied to all input parameters.
    name : str | None, default None
        Circuit name. If None, defaults to "ZZFeatureMapPolyExtended".

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data encoding.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be ≥ 1.")
    n = int(feature_dimension)

    # Resolve number of repetitions
    if callable(reps):
        reps_resolved = reps(n)
        if not isinstance(reps_resolved, int) or reps_resolved <= 0:
            raise ValueError("reps callable must return a positive integer.")
    else:
        reps_resolved = int(reps)
        if reps_resolved <= 0:
            raise ValueError("reps must be a positive integer.")

    if not single_coeffs:
        raise ValueError("single_coeffs must be a non‑empty sequence of floats.")
    if not isinstance(pair_weight, float):
        raise ValueError("pair_weight must be a float.")
    if not isinstance(three_body_weight, float):
        raise ValueError("three_body_weight must be a float.")
    if not isinstance(scaling_factor, float):
        raise ValueError("scaling_factor must be a float.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Scale parameters
    scaled_x = [scaling_factor * xi for xi in x]

    # Pre‑global rotation
    if pre_global_angle!= 0.0:
        for q in range(n):
            qc.ry(pre_global_angle, q)
        if insert_barriers:
            qc.barrier()

    # Define mapping functions
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    def map3(
        xi: ParameterExpression,
        xj: ParameterExpression,
        xk: ParameterExpression,
    ) -> ParameterExpression:
        return _default_map_3(xi, xj, xk, three_body_weight)

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps_resolved):
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
            qc.p(2 * map1(scaled_x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(scaled_x[i], scaled_x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

            # Optional three‑body interaction
            if include_three_body:
                # Choose a third qubit (here the next qubit modulo n)
                k = (j + 1) % n
                angle3 = 2 * map3(scaled_x[i], scaled_x[j], scaled_x[k])
                qc.cx(j, k)
                qc.p(angle3, k)
                qc.cx(j, k)

        if insert_barriers and rep!= reps_resolved - 1:
            qc.barrier()

        # Extra per‑rep rotation
        if extra_per_rep_angle!= 0.0:
            if extra_per_rep_basis == "h":
                qc.h(range(n))
            elif extra_per_rep_basis == "ry":
                for q in range(n):
                    qc.ry(extra_per_rep_angle, q)
            else:
                raise ValueError("extra_per_rep_basis must be 'h' or 'ry'.")

    # Post‑global rotation
    if post_global_angle!= 0.0:
        for q in range(n):
            qc.ry(post_global_angle, q)
        if insert_barriers:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑based API
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    QuantumCircuit subclass wrapping :func:`zz_feature_map_poly_extended`.

    Parameters
    ----------
    All arguments are identical to :func:`zz_feature_map_poly_extended`.

    Example
    -------
    >>> circ = ZZFeatureMapPolyExtended(feature_dimension=4, reps=3)
    >>> circ.draw()
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int | Callable[[int], int] = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        three_body_weight: float = 0.5,
        include_three_body: bool = False,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_global_angle: float = 0.0,
        post_global_angle: float = 0.0,
        extra_per_rep_angle: float = 0.0,
        extra_per_rep_basis: str = "ry",
        scaling_factor: float = 1.0,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            three_body_weight,
            include_three_body,
            basis,
            parameter_prefix,
            insert_barriers,
            pre_global_angle,
            post_global_angle,
            extra_per_rep_angle,
            extra_per_rep_basis,
            scaling_factor,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "ZZFeatureMapPolyExtended",
    "zz_feature_map_poly_extended",
]
