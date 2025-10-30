"""
Extended ZZFeatureMap for quantum kernel methods.
This module provides a functional builder `zz_feature_map_extended` and a
`ZZFeatureMapExtended` subclass that retain the original interface while adding
new hyper‑parameters for higher‑order interactions and optional normalisation.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest‑neighbor pairs
      - ``"circular"``: linear + wrap‑around
      - explicit list of pairs
      - callable returning a sequence of pairs
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


def _default_map_n(x_list: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default φn(x1,…,xn) = ∏(π – xi)."""
    prod = 1
    for xi in x_list:
        prod *= (pi - xi)
    return prod


# --------------------------------------------------------------------------- #
# Extended ZZFeatureMap builder
# --------------------------------------------------------------------------- #

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    include_higher_order: bool = False,
    normalise: bool = False,
    normalisation_range: Tuple[float, float] = (0, pi),
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑feature‑map with optional higher‑order ZZ couplings.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features; must be >= 2.
    reps : int
        Number of repetitions (depth). Default 2.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit entanglement for the pairwise part.
    interaction_order : int
        Order of the ZZ interaction to add (2 for pairwise, 3 for three‑body, etc.).
    include_higher_order : bool
        If ``True`` and ``interaction_order`` > 2, the circuit will include
        n‑body ZZ couplings of the specified order.
    normalise : bool
        If ``True`` a simple linear mapping from the raw data to
        ``normalisation_range`` is applied before the encoding.
        The mapping is performed by the ``data_map_func`` if provided;
        otherwise the default mapping is used.
    normalisation_range : Tuple[float, float]
        Target range for normalisation.  Ignored if ``normalise`` is ``False``.
    data_map_func : Callable | None
        User supplied function mapping a sequence of parameters to a single
        parameter expression.  It is applied to each interaction term.
    parameter_prefix : str
        Prefix for the symbolic parameters.
    insert_barriers : bool
        Insert barriers between repetitions for visual clarity.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if interaction_order < 2:
        raise ValueError("interaction_order must be >= 2.")
    if feature_dimension < interaction_order:
        raise ValueError(
            f"feature_dimension ({feature_dimension}) must be >= interaction_order ({interaction_order})."
        )
    if include_higher_order and interaction_order == 2:
        raise ValueError("include_higher_order is only meaningful for interaction_order > 2.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map_n = _default_map_n
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map_n(x_list: Sequence[ParameterExpression]) -> ParameterExpression:
            return data_map_func(list(x_list))

    pairs = _resolve_entanglement(n, entanglement)

    # Helper to apply higher‑body interaction
    def _apply_n_body(comb: Sequence[int]) -> None:
        target = comb[-1]
        controls = comb[:-1]
        angle = 2 * map_n([x[i] for i in comb])
        for c in controls:
            qc.cx(c, target)
        qc.p(angle, target)
        for c in reversed(controls):
            qc.cx(c, target)

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map_n([x[i], x[j]])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Higher‑order interactions if requested
        if include_higher_order:
            from itertools import combinations
            for comb in combinations(range(n), interaction_order):
                _apply_n_body(comb)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapExtended(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature‑map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        include_higher_order: bool = False,
        normalise: bool = False,
        normalisation_range: Tuple[float, float] = (0, pi),
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            include_higher_order=include_higher_order,
            normalise=normalise,
            normalisation_range=normalisation_range,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
