"""Extended Polynomial ZZFeatureMap with optional triple interactions and pre/post rotations.

This module defines the function `zz_feature_map_poly_extension` and the
class `ZZFeatureMapPolyExtension` that build an enriched feature map
circuit.  The design follows the *extension* scaling paradigm: it adds
higher‑order interactions, adaptive depth, and optional single‑qubit
rotations before and after the main encoding, while keeping the
original structure and parameter binding semantics.

Key features
------------
- **Higher‑order interactions**: pair and/or triple ZZ terms.
- **Adaptive depth**: `reps` can be an integer or a callable that returns
  the number of repetitions based on the feature dimension.
- **Pre/post rotations**: optional RZ rotations on each qubit before and
  after the encoding.
- **Parameter validation**: clear error messages for invalid inputs.
- **Compatibility**: returns a `QuantumCircuit` with an `input_params`
  attribute for easy binding in Qiskit workflows.

Example usage
-------------
>>> from qiskit import QuantumCircuit
>>> from zz_feature_map_poly_extension import zz_feature_map_poly_extension
>>> qc = zz_feature_map_poly_extension(
...     feature_dimension=3,
...     reps=3,
...     interaction_order="both",
...     triple_weight=0.5,
...     pre_rotation=ParameterVector("pre", 3),
...     post_rotation=ParameterVector("post", 3),
... )
>>> qc.draw()
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
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours
      - "circular": linear plus wrap‑around (n‑1,0)
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
# Extended Polynomial ZZ Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extension(
    feature_dimension: int,
    reps: int | Callable[[int], int] = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 0.0,
    interaction_order: str = "pair",  # "pair", "triple", "both"
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    pre_rotation: ParameterVector | None = None,
    post_rotation: ParameterVector | None = None,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended polynomial ZZ feature map with optional triple interactions
    and pre/post RZ rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int or callable
        Number of repetitions of the feature‑map block.  If a callable is
        provided it receives ``feature_dimension`` and must return an int.
    entanglement : str | list[tuple[int, int]] | callable
        Entanglement pattern for pair interactions.
    single_coeffs : sequence[float]
        Coefficients for the polynomial φ1(x) = Σ c_k · x^(k+1).
    pair_weight : float
        Overall scaling factor for the pair interaction φ2(x, y).
    triple_weight : float
        Overall scaling factor for the triple interaction φ3(x, y, z).
    interaction_order : str
        Which interaction terms to include: "pair", "triple", or "both".
    basis : str
        Basis preparation before each repetition: "h" (Hadamard) or "ry"
        (RY(π/2)).
    parameter_prefix : str
        Prefix for the feature parameters.
    pre_rotation : ParameterVector | None
        Optional RZ rotations applied before the basis preparation.
    post_rotation : ParameterVector | None
        Optional RZ rotations applied after all repetitions.
    insert_barriers : bool
        Insert barriers between logical blocks for visual clarity.
    name : str | None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The attribute ``input_params``
        holds the feature parameters for binding.

    Raises
    ------
    ValueError
        If input arguments are inconsistent (e.g., insufficient qubits for
        triple interactions).
    """
    # Validate feature dimension
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    # Resolve repetitions
    if callable(reps):
        reps_val = int(reps(feature_dimension))
    else:
        reps_val = int(reps)
    if reps_val < 1:
        raise ValueError("reps must be >= 1.")

    # Validate interaction order
    if interaction_order not in {"pair", "triple", "both"}:
        raise ValueError("interaction_order must be 'pair', 'triple', or 'both'.")

    if interaction_order in {"triple", "both"} and n < 3:
        raise ValueError("At least 3 qubits required for triple interactions.")

    # Build circuit
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtension")

    # Feature parameters
    x = ParameterVector(parameter_prefix, n)

    # Optional pre‑rotations
    if pre_rotation is not None:
        if len(pre_rotation)!= n:
            raise ValueError("pre_rotation ParameterVector length must match feature_dimension.")
        for i in range(n):
            qc.rz(pre_rotation[i], i)

    # Resolve pair entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Resolve triple combinations
    triples: List[Tuple[int, int, int]] = []
    if interaction_order in {"triple", "both"}:
        triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    # Define mapping functions
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr += c * p
            p = p * xi  # next power
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return triple_weight * xi * xj * xk

    # Main repetition loop
    for rep in range(reps_val):
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

        # Pair ZZ interactions
        if interaction_order in {"pair", "both"}:
            for (i, j) in pairs:
                angle = 2 * map2(x[i], x[j])
                qc.cx(i, j)
                qc.p(angle, j)
                qc.cx(i, j)

        # Triple ZZ interactions
        if interaction_order in {"triple", "both"}:
            for (i, j, k) in triples:
                angle = 2 * map3(x[i], x[j], x[k])
                # Implement exp(i θ Z_i Z_j Z_k) using two CNOTs and a Z‑rotation
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, j)

        if insert_barriers and rep!= reps_val - 1:
            qc.barrier()

    # Optional post‑rotations
    if post_rotation is not None:
        if len(post_rotation)!= n:
            raise ValueError("post_rotation ParameterVector length must match feature_dimension.")
        for i in range(n):
            qc.rz(post_rotation[i], i)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtension(QuantumCircuit):
    """Class‑style wrapper for zz_feature_map_poly_extension.

    The constructor accepts the same keyword arguments as the functional
    interface and composes the built circuit into the new instance.  The
    resulting object behaves like a normal `QuantumCircuit` and exposes
    the `input_params` attribute for parameter binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int | Callable[[int], int] = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 0.0,
        interaction_order: str = "pair",
        basis: str = "h",
        parameter_prefix: str = "x",
        pre_rotation: ParameterVector | None = None,
        post_rotation: ParameterVector | None = None,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyExtension",
    ) -> None:
        built = zz_feature_map_poly_extension(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triple_weight,
            interaction_order,
            basis,
            parameter_prefix,
            pre_rotation,
            post_rotation,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtension", "zz_feature_map_poly_extension"]
