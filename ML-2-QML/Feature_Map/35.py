"""Controlled‑modified ZZFeatureMapPoly: symmetric pair interactions and shared parameters."""

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
      - "linear": nearest neighbours (0,1), (1,2), …
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Controlled‑modified polynomial ZZ feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float | Sequence[float] = 1.0,
    shared_pair_weight: bool = True,
    pair_function: Callable[[ParameterExpression, ParameterExpression], ParameterExpression] | None = None,
    data_transform: Callable[[ParameterExpression], ParameterExpression] | None = None,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modified polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int
        Number of repetitions of the feature‑map block.
    entanglement : str or sequence or callable
        Entanglement pattern. See ``_resolve_entanglement`` for details.
    single_coeffs : sequence of float
        Coefficients for the polynomial φ1(x) = Σ_k coeff_k · x^{k+1}.
    pair_weight : float or sequence of float
        Weight(s) for the pair function φ2. If ``shared_pair_weight`` is True, a single
        weight is used for all pairs; otherwise a sequence matching the number of pairs
        is required.
    shared_pair_weight : bool
        If True, ``pair_weight`` is a scalar shared across all pair interactions.
    pair_function : callable
        Function ``f(xi, xj)`` returning a parameter expression for the pair interaction.
        Defaults to ``lambda xi, xj: pair_weight * xi * xj``.
    data_transform : callable
        Optional transformation applied to each feature before mapping. Should accept
        a ``ParameterExpression`` and return a ``ParameterExpression``.
    basis : str
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str
        Prefix for the parameter vector.
    insert_barriers : bool
        Whether to insert barriers between layers.
    name : str or None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    - The circuit exposes ``.input_params`` for binding.
    - All parameters are ``ParameterExpression`` objects from a ``ParameterVector``.
    - Error messages validate feature dimension, entanglement, and weight consistency.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for data
    x = ParameterVector(parameter_prefix, n)

    # Optional data transform
    def _transform(expr: ParameterExpression) -> ParameterExpression:
        return data_transform(expr) if data_transform is not None else expr

    # Build φ1
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = _transform(xi)
        for c in single_coeffs:
            expr = expr + c * p
            p = p * _transform(xi)  # next power
        return expr

    # Build φ2
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        if pair_function is not None:
            return pair_function(_transform(xi), _transform(xj))
        # default: weighted product
        if shared_pair_weight:
            weight_expr = pair_weight
        else:
            # pair_weight will be indexed later
            weight_expr = 0  # placeholder, will be replaced per pair
        return weight_expr * _transform(xi) * _transform(xj)

    pairs = _resolve_entanglement(n, entanglement)

    # Validate pair_weight if not shared
    if not shared_pair_weight:
        if isinstance(pair_weight, (float, int)):
            raise ValueError("pair_weight must be a sequence when shared_pair_weight is False.")
        if len(pair_weight)!= len(pairs):
            raise ValueError(
                f"pair_weight length {len(pair_weight)} does not match number of pairs {len(pairs)}."
            )

    # Main loop
    for rep in range(int(reps)):
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

        # ZZ via CX–P–CX
        for idx, (i, j) in enumerate(pairs):
            # Determine weight
            if shared_pair_weight:
                weight = pair_weight
            else:
                weight = pair_weight[idx]
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified polynomial ZZ feature map.

    The constructor mirrors the functional API.  The circuit exposes
    ``.input_params`` for parameter binding.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int
        Number of repetitions of the feature‑map block.
    entanglement : str or sequence or callable
        Entanglement pattern. See ``_resolve_entanglement`` for details.
    single_coeffs : sequence of float
        Coefficients for the polynomial φ1.
    pair_weight : float or sequence of float
        Weight(s) for the pair function φ2.
    shared_pair_weight : bool
        If True, ``pair_weight`` is a scalar shared across all pair interactions.
    pair_function : callable
        Function ``f(xi, xj)`` returning a parameter expression for the pair interaction.
    data_transform : callable
        Optional transformation applied to each feature before mapping.
    basis : str
        Basis preparation: ``"h"`` or ``"ry"``.
    parameter_prefix : str
        Prefix for the parameter vector.
    insert_barriers : bool
        Whether to insert barriers between layers.
    name : str
        Circuit name.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float | Sequence[float] = 1.0,
        shared_pair_weight: bool = True,
        pair_function: Callable[[ParameterExpression, ParameterExpression], ParameterExpression] | None = None,
        data_transform: Callable[[ParameterExpression], ParameterExpression] | None = None,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            shared_pair_weight,
            pair_function,
            data_transform,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
