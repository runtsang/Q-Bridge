"""ZZFeatureMapRZZExtension – an extended feature‑map using native RZZ gates with higher‑order terms.

Author: Quantum‑AI‑Architect
The following module implements two entry‑points – a functional wrapper and a class‑based wrapper.
"""

from __future__ import annotations

import math
from math import pi
from typing import Callable, List, Sequence, Tuple, Optional
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbors (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
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
) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


def _compute_higher_order_sum(
    x: List[ParameterExpression],
    order: int,
) -> List[ParameterExpression]:
    """Return a list of higher‑order interaction terms for each qubit.

    For ``order`` == 3, each term is the sum over all products
    involving the target qubit and any two distinct other qubits.
    """
    n = len(x)
    sums: List[ParameterExpression] = [0 for _ in range(n)]
    if order < 3:
        return sums
    for i in range(n):
        # iterate over all unordered pairs (j, k) with j < k, j!= i, k!= i
        for j, k in combinations([idx for idx in range(n) if idx!= i], 2):
            sums[i] += x[i] * x[j] * x[k]
    return sums


# --------------------------------------------------------------------------- #
#  Feature‑map implementation
# --------------------------------------------------------------------------- #

def zz_feature_map_rzz_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    interaction_order: int = 2,
    data_rescale: float = 1.0,
    activation_layer: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended ZZ feature map using native ``rzz`` gates with optional higher‑order
    interaction layers, data rescaling, and a lightweight activation circuit.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (and qubits). Must be >= 2.
    reps : int, default 2
        Number of feature‑map repetitions.
    entanglement : str | sequence | callable, default "full"
        Entanglement pattern. See :func:`_resolve_entanglement`.
    data_map_func : callable, optional
        User‑supplied mapping from a sequence of parameters to a single
        parameter expression. If ``None`` defaults to the canonical
        ``φ1``/``φ2``/``φ3`` functions.
    parameter_prefix : str, default "x"
        Prefix for the ``ParameterVector`` names.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for visual clarity.
    pair_scale : float, default 1.0
        Scaling factor applied to all pairwise interaction angles.
    interaction_order : int, default 2
        Minimum interaction order to include. Valid values: 2 or 3.
    data_rescale : float, default 1.0
        Global scaling applied to the raw data before mapping.
    activation_layer : bool, default False
        If ``True`` appends a simple activation circuit consisting of
        ``p`` rotations that depend on the sum of all features.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if pair_scale <= 0:
        raise ValueError("pair_scale must be positive.")
    if data_rescale <= 0:
        raise ValueError("data_rescale must be positive.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtension")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Default mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - xi) * (pi - xj)

        def map3(
            xi: ParameterExpression,
            xj: ParameterExpression,
            xk: ParameterExpression,
        ) -> ParameterExpression:
            return (pi - xi) * (pi - xj) * (pi - xk)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(
            xi: ParameterExpression,
            xj: ParameterExpression,
            xk: ParameterExpression,
        ) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    pairs = _resolve_entanglement(n, entanglement)

    # Pre‑compute higher‑order sums if required
    higher_order_terms: List[ParameterExpression] = [0 for _ in range(n)]
    if interaction_order == 3:
        higher_order_terms = _compute_higher_order_sum(x, 3)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise RZZ entanglement
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)

        # Higher‑order interactions (order 3)
        if interaction_order == 3:
            for i in range(n):
                qc.p(2 * pair_scale * higher_order_terms[i], i)

        # Optional activation layer
        if activation_layer:
            total_sum = sum(x)
            for i in range(n):
                qc.p(data_rescale * total_sum, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZExtension(QuantumCircuit):
    """Class‑style wrapper for the extended RZZ‑entangled feature map.

    The constructor builds the circuit via :func:`zz_feature_map_rzz_extension`
    and then composes it into the current instance.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        interaction_order: int = 2,
        data_rescale: float = 1.0,
        activation_layer: bool = False,
        name: str = "ZZFeatureMapRZZExtension",
    ) -> None:
        built = zz_feature_map_rzz_extension(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            interaction_order,
            data_rescale,
            activation_layer,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtension", "zz_feature_map_rzz_extension"]
