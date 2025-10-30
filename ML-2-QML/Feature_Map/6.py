"""Extended Polynomial ZZ Feature Map with optional three‑qubit interactions and
pre/post‑rotations.

This module provides a functional interface (`zz_feature_map_poly_extended`)
and an object‑oriented wrapper (`ZZFeatureMapPolyExtended`) that construct a
parameterised quantum circuit suitable for variational quantum algorithms.
The design extends the base polynomial ZZ feature map by adding:

* optional three‑qubit ZZ interactions (full‑connectivity only)
* configurable pre‑ and post‑rotations in either Hadamard or RY(π/2) basis
* feature‑vector normalisation (1/√n scaling)
* barrier insertion for clearer visualisation
* comprehensive error handling and input validation

Supported datasets
------------------
The circuit accepts any real‑valued feature vector of length equal to the
number of qubits.  Data is mapped to rotation angles via polynomial
coefficients for single‑qubit terms and linear or cubic products for
pairwise and three‑qubit interactions.

Parameter constraints
---------------------
* `feature_dimension` ≥ 2 (≥3 if `triple_weight` > 0)
* `reps` ≥ 1
* `entanglement` must be `"full"`, `"linear"`, `"circular"`, or a custom list
  of pairs for two‑qubit couplings.  Three‑qubit couplings are only supported
  for `"full"`.
* `single_coeffs` may contain any number of coefficients; higher orders are
  interpreted as higher‑power polynomial terms.
* `basis`, `pre_basis`, and `post_basis` must be `"h"` or `"ry"` if provided.
* `parameter_prefix` must be a string suitable for Qiskit ParameterVector.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable, Iterable, List, Sequence, Tuple

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
      - `"full"`: all‑to‑all pairs (i < j)
      - `"linear"`: nearest neighbors
      - `"circular"`: linear + wrap‑around
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


def _resolve_triples(num_qubits: int, entanglement: str) -> List[Tuple[int, int, int]]:
    """Return all unique triples for full‑connectivity."""
    if entanglement!= "full":
        raise ValueError("Three‑qubit interactions are only supported with 'full' entanglement.")
    return [(i, j, k) for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            for k in range(j + 1, num_qubits)]


# ---------------------------------------------------------------------------
# Extended Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 0.0,
    basis: str = "h",  # "h" or "ry"
    pre_basis: str | None = None,
    post_basis: str | None = None,
    parameter_prefix: str = "x",
    normalize: bool = False,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with optional three‑qubit interactions and
    configurable pre/post‑rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / length of the classical feature vector.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | sequence | callable, default "full"
        Specification of two‑qubit coupling pairs.
    single_coeffs : sequence of float, default (1.0,)
        Coefficients for the polynomial map φ₁(x) = Σ c_k · x^{k+1}.
    pair_weight : float, default 1.0
        Weight for the two‑qubit interaction term φ₂(x, y) = x·y.
    triple_weight : float, default 0.0
        Weight for the three‑qubit interaction term φ₃(x, y, z) = x·y·z.
    basis : str, default "h"
        Basis preparation before each repetition ("h" = Hadamard, "ry" = RY(π/2)).
    pre_basis : str | None, default None
        Optional rotation applied before the main basis preparation.
    post_basis : str | None, default None
        Optional rotation applied after the main basis preparation.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    normalize : bool, default False
        If True, scale each feature component by 1/√n before mapping.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for visual clarity.
    name : str | None, default None
        Name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for data encoding.

    Raises
    ------
    ValueError
        If input arguments are inconsistent or unsupported.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if triple_weight!= 0.0 and feature_dimension < 3:
        raise ValueError("triple_weight > 0 requires at least 3 qubits.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")
    if pre_basis is not None and pre_basis not in ("h", "ry"):
        raise ValueError("pre_basis must be None, 'h', or 'ry'.")
    if post_basis is not None and post_basis not in ("h", "ry"):
        raise ValueError("post_basis must be None, 'h', or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")
    x = ParameterVector(parameter_prefix, n)

    # scaling factor for optional normalisation
    scale = 1 / sqrt(n) if normalize else 1.0

    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = scale * xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi  # next power of xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * scale * scale * xi * xj

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return triple_weight * scale * scale * scale * xi * xj * xk

    pair_pairs = _resolve_entanglement(n, entanglement)
    triple_triples = _resolve_triples(n, entanglement) if triple_weight!= 0.0 else []

    for rep in range(int(reps)):
        # optional pre‑rotation
        if pre_basis == "h":
            qc.h(range(n))
        elif pre_basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)

        # main basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # "ry"
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # two‑qubit ZZ interactions
        for (i, j) in pair_pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # three‑qubit ZZ interactions (if requested)
        for (i, j, k) in triple_triples:
            angle = 2 * map3(x[i], x[j], x[k])
            qc.cx(i, j)
            qc.cx(i, k)
            qc.p(angle, k)
            qc.cx(i, k)
            qc.cx(i, j)

        # optional post‑rotation
        if post_basis == "h":
            qc.h(range(n))
        elif post_basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Object‑oriented wrapper for the polynomial ZZ feature map with optional
    three‑qubit interactions and configurable pre/post‑rotations.

    The constructor simply builds the circuit via
    :func:`zz_feature_map_poly_extended` and composes it into the object.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 0.0,
        basis: str = "h",
        pre_basis: str | None = None,
        post_basis: str | None = None,
        parameter_prefix: str = "x",
        normalize: bool = False,
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
            basis,
            pre_basis,
            post_basis,
            parameter_prefix,
            normalize,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
