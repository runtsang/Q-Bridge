"""Polynomial ZZFeatureMap variant with higher‑order interactions and optional pre/post rotations."""
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
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2),...
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


def _apply_pre_rotation(qc: QuantumCircuit, pre: str | None, num_qubits: int) -> None:
    """Apply a uniform pre‑rotation across all qubits."""
    if pre is None or pre.lower() == "none":
        return
    if pre.lower() == "ry":
        for q in range(num_qubits):
            qc.ry(pi / 2, q)
    elif pre.lower() == "rx":
        for q in range(num_qubits):
            qc.rx(pi / 2, q)
    elif pre.lower() == "rz":
        for q in range(num_qubits):
            qc.rz(pi / 2, q)
    else:
        raise ValueError(f"Unsupported pre‑rotation '{pre}'. Use 'ry', 'rx', 'rz', or None.")


def _apply_post_rotation(qc: QuantumCircuit, post: str | None, num_qubits: int) -> None:
    """Apply a uniform post‑rotation across all qubits."""
    if post is None or post.lower() == "none":
        return
    if post.lower() == "ry":
        for q in range(num_qubits):
            qc.ry(pi / 2, q)
    elif post.lower() == "rx":
        for q in range(num_qubits):
            qc.rx(pi / 2, q)
    elif post.lower() == "rz":
        for q in range(num_qubits):
            qc.rz(pi / 2, q)
    else:
        raise ValueError(f"Unsupported post‑rotation '{post}'. Use 'ry', 'rx', 'rz', or None.")


# ---------------------------------------------------------------------------
# Extended Polynomial ZZ Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int | str = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 1.0,
    include_triple: bool = False,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotation: str | None = None,
    post_rotation: str | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits. Must be >= 2.
    reps : int | str, default 2
        Number of repetitions. If *'auto'* is passed, depth is set to
        ``max(2, feature_dimension // 4)``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. See :func:`_resolve_entanglement`.
    single_coeffs : Sequence[float]
        Coefficients for the polynomial φ1(x) = Σ c_k x^{k+1}.
    pair_weight : float
        Scaling factor for the pair interaction φ2(x, y).
    triple_weight : float
        Scaling factor for the triple interaction φ3(x, y, z). Effective only if
        ``include_triple`` is True.
    include_triple : bool
        Enable triple‑qubit ZZ interactions.
    basis : str, default 'h'
        Basis preparation: 'h' for Hadamard, 'ry' for RY(π/2).
    parameter_prefix : str
        Prefix for the ParameterVector name.
    insert_barriers : bool
        Insert barrier after each major block for visual clarity.
    pre_rotation : str | None
        Optional uniform rotation before basis preparation. Options: 'ry', 'rx',
        'rz', or None.
    post_rotation : str | None
        Optional uniform rotation after entanglement. Options: 'ry', 'rx',
        'rz', or None.
    name : str | None
        Circuit name. Defaults to ``"ZZFeatureMapPolyExtended"``.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a classical feature vector.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    # Resolve repetitions
    if isinstance(reps, str):
        if reps.lower() == "auto":
            reps = max(2, n // 4)
        else:
            raise ValueError(f"Unrecognised reps string: {reps!r}")
    reps = int(reps)
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Polynomial map for single qubits
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi  # next power
        return expr

    # Pair interaction map
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    # Triple interaction map
    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return triple_weight * xi * xj * xk

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps):
        # Optional pre‑rotation
        _apply_pre_rotation(qc, pre_rotation, n)

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

        # Pair‑qubit ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional triple‑qubit ZZ interactions
        if include_triple:
            # Use all distinct triples
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        angle = 2 * map3(x[i], x[j], x[k])
                        # Implement via controlled‑controlled‑phase using two CXs
                        qc.ccx(i, j, k)
                        qc.p(angle, k)
                        qc.ccx(i, j, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

        # Optional post‑rotation
        _apply_post_rotation(qc, post_rotation, n)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Class‑style wrapper for the extended polynomial ZZ feature map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int | str = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 1.0,
        include_triple: bool = False,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotation: str | None = None,
        post_rotation: str | None = None,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triple_weight,
            include_triple,
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


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
