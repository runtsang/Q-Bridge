"""Extended Polynomial ZZFeatureMap with optional higher‑order interactions and normalisation."""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest‑neighbor pairs (i, i+1)
      - ``"circular"``: linear + wrap‑around (n-1, 0)
      - explicit list of pairs [(i, j), …]
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
        return [(int(i), int(j)) for (i, j) in entanglement(num_qubits)]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs

def _resolve_triples(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Return all distinct triples (i, j, k) with i < j < k."""
    from itertools import combinations
    return list(combinations(range(num_qubits), 3))

# --------------------------------------------------------------------------- #
# Feature map
# --------------------------------------------------------------------------- #

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 0.0,
    basis_pre: str = "h",
    basis_post: str = "none",
    normalisation: Callable[[Sequence[float]], Sequence[float]] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with optional higher‑order interactions, pre‑ and post‑rotations,
    and user‑supplied normalisation.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of pairwise entanglement.
    single_coeffs : Sequence[float]
        Coefficients for the polynomial φ₁(x) = Σ c_k · x^{k+1}.
    pair_weight : float
        Weight for pairwise interaction φ₂(x, y) = x · y.
    triple_weight : float
        Weight for triple interaction φ₃(x, y, z) = x · y · z.
    basis_pre : str
        Basis rotation applied before encoding; options are ``"h"``, ``"ry"``, or ``"none"``.
    basis_post : str
        Optional basis rotation applied after encoding; same options as ``basis_pre``.
    normalisation : Callable[[Sequence[float]], Sequence[float]] | None
        Function applied to the input feature vector before binding.
    parameter_prefix : str
        Prefix for the symbolic parameters.
    insert_barriers : bool
        Insert barriers between logical sections for visualization.
    name : str | None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a feature vector.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if basis_pre not in {"h", "ry", "none"}:
        raise ValueError("basis_pre must be 'h', 'ry', or 'none'.")
    if basis_post not in {"h", "ry", "none"}:
        raise ValueError("basis_post must be 'h', 'ry', or 'none'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Symbolic parameters
    x = ParameterVector(parameter_prefix, n)

    # Helper maps
    def poly_map(xi: ParameterExpression) -> ParameterExpression:
        """Compute φ₁(x) = Σ c_k · x^{k+1}."""
        expr: ParameterExpression = 0
        power: ParameterExpression = xi
        for coeff in single_coeffs:
            expr += coeff * power
            power *= xi
        return expr

    def pair_map(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Compute φ₂(x, y) = x · y."""
        return pair_weight * xi * xj

    def triple_map(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        """Compute φ₃(x, y, z) = x · y · z."""
        return triple_weight * xi * xj * xk

    pairs = _resolve_entanglement(n, entanglement)
    triples = _resolve_triples(n) if triple_weight!= 0.0 else []

    for rep in range(int(reps)):
        # Pre‑rotation
        if basis_pre == "h":
            qc.h(range(n))
        elif basis_pre == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * poly_map(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * pair_map(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Triple‑interaction via CX‑CX–RZ–CX‑CX
        for (i, j, k) in triples:
            angle = 2 * triple_map(x[i], x[j], x[k])
            # Controlled RZ on k with controls i and j
            qc.cx(i, j)
            qc.cx(j, k)
            qc.p(angle, k)
            qc.cx(j, k)
            qc.cx(i, j)

        # Post‑rotation
        if basis_post == "h":
            qc.h(range(n))
        elif basis_post == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    qc.normalisation = normalisation  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    Object‑oriented wrapper for :func:`zz_feature_map_poly_extended`.

    The class inherits from :class:`~qiskit.circuit.QuantumCircuit` and
    exposes the same parameters as the functional form.  It also keeps a
    reference to the original :class:`ParameterVector` via ``input_params``.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 0.0,
        basis_pre: str = "h",
        basis_post: str = "none",
        normalisation: Callable[[Sequence[float]], Sequence[float]] | None = None,
        parameter_prefix: str = "x",
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
            basis_pre,
            basis_post,
            normalisation,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.normalisation = built.normalisation  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
