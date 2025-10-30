"""Extended polynomial ZZFeatureMap with higher‑order interactions and optional rotations.

The original `zz_feature_map_poly` is enriched with:
  * **Triplet interactions** – optional CCZ‑style phases for all qubit triples.
  * **Pre‑ and post‑rotations** – additional RY(π/2) gates before and after the main encoding.
  * **Flexible entanglement** – same spec as the seed but with an optional adaptive pattern.
  * **Parameter validation** – clear error messages for invalid inputs.
  * **Compatibility** – exposes both a helper function and a QuantumCircuit subclass, retains `input_params` for Qiskit data encoding workflows.

The feature map remains fully parameterised and can be bound to classical data vectors of length `feature_dimension`.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      * `"full"`      – all‑to‑all pairs (i < j)
      * `"linear"`    – nearest neighbours (0,1), (1,2), …
      * `"circular"`  – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs like `[(0, 2), (1, 3)]`
      * `callable`    – f(num_qubits) -> sequence of (i, j)

    Raises:
        ValueError: if the spec is unknown or contains invalid pairs.
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


def _resolve_triplet_entanglement(num_qubits: int) -> list[Tuple[int, int, int]]:
    """Return all unique qubit triples (i < j < k)."""
    return [(i, j, k)
            for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            for k in range(j + 1, num_qubits)]


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Extended Polynomial ZZ Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triplet_weight: float = 0.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    r"""Extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / length of the classical feature vector.
    reps : int, default 2
        Number of feature‑map repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement specification (see ``_resolve_entanglement``).
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial map φ1(x) = Σ_k coeff_k · x^{k+1}.
    pair_weight : float, default 1.0
        Scaling factor for pair‑phase φ2(x, y) = weight · x · y.
    triplet_weight : float, default 0.0
        Scaling factor for triplet‑phase φ3(x, y, z) = weight · x · y · z.
        If zero, triplet interactions are omitted.
    basis : str, default "h"
        Basis preparation: `"h"` for Hadamards, `"ry"` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    pre_rotation : bool, default False
        Apply an extra RY(π/2) on all qubits before the main basis preparation.
    post_rotation : bool, default False
        Apply an extra RY(π/2) on all qubits after entanglement.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit ready for data encoding.

    Notes
    -----
    * The circuit exposes ``qc.input_params`` for Qiskit data‑encoding workflows.
    * Triplet interactions are implemented via a CCZ‑style decomposition:
      CX(i,k); CX(j,k); P(angle) on k; CX(j,k); CX(i,k).
    * Parameter binding follows the order of the ``ParameterVector``.

    Raises
    ------
    ValueError
        For invalid `feature_dimension`, `entanglement` spec, or negative weights.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if pair_weight < 0:
        raise ValueError("pair_weight must be non‑negative.")
    if triplet_weight < 0:
        raise ValueError("triplet_weight must be non‑negative.")
    if not single_coeffs:
        raise ValueError("single_coeffs must contain at least one coefficient.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtension")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Polynomial map for single‑qubit phases
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi  # next power
        return expr

    # Pair‑phase map
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    # Triplet‑phase map
    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return triplet_weight * xi * xj * xk

    pairs = _resolve_entanglement(n, entanglement)
    triplets = _resolve_triplet_entanglement(n) if triplet_weight > 0 else []

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for q in range(n):
                qc.ry(pi / 2, q)

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
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pair‑phase interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional triplet‑phase interactions via CCZ‑style decomposition
        if triplets:
            for (i, j, k) in triplets:
                angle = 2 * map3(x[i], x[j], x[k])
                qc.cx(i, k)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

        # Optional post‑rotation
        if post_rotation:
            for q in range(n):
                qc.ry(pi / 2, q)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtension(QuantumCircuit):
    """Class‑style wrapper for the extended polynomial ZZ feature map.

    The constructor accepts the same arguments as ``zz_feature_map_poly_extension``.
    The resulting circuit is built and composed into the instance.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triplet_weight: float = 0.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        name: str = "ZZFeatureMapPolyExtension",
    ) -> None:
        built = zz_feature_map_poly_extension(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triplet_weight,
            basis,
            parameter_prefix,
            insert_barriers,
            pre_rotation,
            post_rotation,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtension", "zz_feature_map_poly_extension"]
