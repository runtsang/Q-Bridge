"""Extended ZZFeatureMap with higher‑order interactions and optional rotations.

This module implements an enriched encoding circuit that extends the
canonical ZZFeatureMap.  The extension adds:
* **Triplet (3‑qubit) interactions** for richer correlation capture.
* **Pre‑ and post‑rotations** (Rx) that can be toggled per repetition.
* **Data scaling** to normalize or amplify input features.
* **Adaptive depth** support via an optional `reps` argument.
* Compatibility with Qiskit’s parameter binding workflow.

The functional interface `zz_feature_map_extended` and the OO wrapper
`ZZFeatureMapExtended` expose identical behavior.

Author: Quantum Feature Map Architect
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to the entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours
      - ``"circular"``: linear plus wrap‑around
      - explicit list of pairs ``[(0, 2), (1, 3)]``
      - callable ``f(n) -> sequence``

    Raises
    ------
    ValueError
        If an unknown spec or invalid pair is provided.
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


def _generate_triplet_pairs(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Return all unique 3‑qubit combinations for triplet entanglement."""
    return [
        (i, j, k)
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
        for k in range(j + 1, num_qubits)
    ]


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Functional feature‑map builder
# ---------------------------------------------------------------------------

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int | None = None,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    data_scaling: float = 1.0,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension
        Number of classical features / qubits. Must be >= ``interaction_order``.
    reps
        Number of repetitions (layers). If ``None`` defaults to 2.
    entanglement
        Specification of two‑qubit entanglement pattern.
    interaction_order
        1 → single‑qubit phases only
        2 → pairwise ZZ interactions (default)
        3 → add triplet (3‑qubit) ZZ interactions
    data_map_func
        User‑supplied mapping from a list of feature parameters to a single
        parameter expression. If ``None`` the default quadratic or cubic
        functions are used.
    data_scaling
        Multiplicative factor applied to all input features before mapping.
    pre_rotation
        If ``True`` apply an Rx(2·φ1) rotation immediately after the
        Hadamard on each qubit.
    post_rotation
        If ``True`` apply an Rx(2·φ1) rotation after all entanglements
        and before the next repetition.
    parameter_prefix
        Prefix for the parameter vector names.
    insert_barriers
        Insert barriers between layers for visual clarity.
    name
        Optional circuit name; defaults to ``"ZZFeatureMapExtended"``.

    Returns
    -------
    QuantumCircuit
        The parameterised encoding circuit.

    Raises
    ------
    ValueError
        If feature_dimension < interaction_order or invalid parameters.
    """
    if feature_dimension < interaction_order:
        raise ValueError(
            "feature_dimension must be >= interaction_order "
            f"(got {feature_dimension} < {interaction_order})."
        )

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    x = ParameterVector(parameter_prefix, n)

    # Resolve mapping functions with optional scaling
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:

        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_scaling * data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_scaling * data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_scaling * data_map_func([xi, xj, xk])

    # Prepare entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)
    triplet_pairs = _generate_triplet_pairs(n) if interaction_order >= 3 else []

    # Default repetition count
    reps = int(reps) if reps is not None else 2

    for rep in range(reps):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.rx(2 * map1(x[i]), i)

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        # Two‑qubit ZZ entanglement via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Optional three‑qubit entanglement via CCX–P–CCX
        if interaction_order >= 3:
            for (i, j, k) in triplet_pairs:
                angle_3 = 2 * map3(x[i], x[j], x[k])
                qc.ccx(i, j, k)
                qc.p(angle_3, k)
                qc.ccx(i, j, k)

        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.rx(2 * map1(x[i]), i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapExtended(QuantumCircuit):
    """Object‑oriented wrapper for the extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension
        Number of classical features / qubits.
    reps
        Number of repetitions (layers). If ``None`` defaults to 2.
    entanglement
        Specification of two‑qubit entanglement pattern.
    interaction_order
        1 → single‑qubit phases only
        2 → pairwise ZZ interactions (default)
        3 → add triplet (3‑qubit) ZZ interactions
    data_map_func
        User‑supplied mapping from a list of feature parameters to a single
        parameter expression. If ``None`` the default quadratic or cubic
        functions are used.
    data_scaling
        Multiplicative factor applied to all input features before mapping.
    pre_rotation
        If ``True`` apply an Rx(2·φ1) rotation immediately after the
        Hadamard on each qubit.
    post_rotation
        If ``True`` apply an Rx(2·φ1) rotation after all entanglements
        and before the next repetition.
    parameter_prefix
        Prefix for the parameter vector names.
    insert_barriers
        Insert barriers between layers for visual clarity.
    name
        Optional circuit name; defaults to ``"ZZFeatureMapExtended"``.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int | None = None,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        data_scaling: float = 1.0,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            data_scaling=data_scaling,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
