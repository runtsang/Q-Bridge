"""Extended ZZ-feature-map with optional higher-order interactions, pre/post rotations, and feature normalisation.

The module exposes both a functional builder `zz_feature_map_extended` and an OO wrapper `ZZFeatureMapExtended`.  The interface mirrors the original `ZZFeatureMap` while adding the following extensions:

* **Interaction order** – support 2‑body (ZZ) and 3‑body (CCX‑P‑CCX) entanglers.
* **Pre‑ and post‑rotations** – user‑defined single‑qubit gates applied before/after each repetition.
* **Feature normalisation** – optional linear scaling of the input vector.
* **Custom data mapping** – replace the default φ₁, φ₂, φ₃ functions with arbitrary expressions.

The circuit remains compatible with Qiskit's data‑encoding workflow: a `ParameterVector` named *x* is exposed as `input_params` and can be bound with `bind_parameters`.

The implementation validates all arguments, raises informative errors, and inserts optional barriers for debugging.
"""

from __future__ import annotations

import itertools
from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.library import CCXGate

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
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear + wrap‑around (n‑1,0) if n > 2
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
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _resolve_interactions(
    num_qubits: int,
    order: int,
) -> List[Tuple[int,...]]:
    """Return all qubit tuples for the requested interaction order.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    order : int
        Interaction order (>=2).  For 2 the result is pairs; for 3 triples.

    Raises
    ------
    ValueError
        If ``order`` is less than 2 or larger than ``num_qubits``.
    """
    if order < 2:
        raise ValueError("Interaction order must be >= 2.")
    if order > num_qubits:
        raise ValueError(f"Interaction order {order} cannot exceed number of qubits {num_qubits}.")
    return list(itertools.combinations(range(num_qubits), order))


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ₂(x, y) = (π – x)(π – y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ₃(x, y, z) = (π – x)(π – y)(π – z)."""
    return (pi - x) * (pi - y) * (pi - z)


# --------------------------------------------------------------------------- #
# Functional builder
# --------------------------------------------------------------------------- #

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    pre_rotations: Sequence[Tuple[str, int, ParameterExpression]] | None = None,
    post_rotations: Sequence[Tuple[str, int, ParameterExpression]] | None = None,
    normalisation: float | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑feature‑map.

    The circuit follows the canonical structure

        H → P(2·φ₁) → (CX–P(2·φ₂)–CX) → (CCX–P(2·φ₃)–CCX) → …

    and optionally inserts user‑defined pre‑ and post‑rotations.  Feature values are
    linearly scaled by ``normalisation`` before being passed to the data‑mapping
    functions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.  Must be >= 2.
    reps : int, default 2
        Number of repetitions (depth).
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of which qubit pairs receive 2‑body entanglers.
    interaction_order : int, default 2
        Highest interaction order to include.  Supported values: 2 (default) and 3.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        User‑supplied mapping from a list of parameters to a single phase expression.
        If ``None`` the default φ₁/φ₂/φ₃ are used.
    pre_rotations : Sequence[Tuple[str, int, ParameterExpression]] | None
        List of (gate_name, qubit_index, angle_expr) applied before each repetition.
        Supported gate_name are ``"rx"``, ``"ry"``, ``"rz"``, ``"p"``.
    post_rotations : Sequence[Tuple[str, int, ParameterExpression]] | None
        Same as ``pre_rotations`` but applied after each repetition.
    normalisation : float | None
        Optional scaling factor applied to all input features before mapping.
    parameter_prefix : str, default "x"
        Prefix used for parameter names.
    insert_barriers : bool, default False
        Insert barriers after each major block for visual clarity.
    name : str | None
        Optional circuit name.  Defaults to ``"ZZFeatureMapExtended"``.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding and execution.

    Raises
    ------
    ValueError
        If any argument is out of bounds or incompatible.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapExtended.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3 (higher orders not supported).")
    if normalisation is not None and normalisation <= 0:
        raise ValueError("normalisation must be positive if supplied.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Parameter vector for the raw input features
    x = ParameterVector(parameter_prefix, n)

    # Apply optional scaling to the raw parameters
    if normalisation is not None:
        x_scaled = [xi * normalisation for xi in x]
    else:
        x_scaled = list(x)  # type: ignore[assignment]

    # Resolve entanglement pairs and higher‑order interactions
    pairs = _resolve_entanglement(n, entanglement)
    triples = _resolve_interactions(n, 3) if interaction_order == 3 else []

    # Prepare mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    # Validate rotation lists
    def _validate_rotations(rotations: Sequence[Tuple[str, int, ParameterExpression]] | None) -> List[Tuple[str, int, ParameterExpression]]:
        if rotations is None:
            return []
        allowed = {"rx", "ry", "rz", "p"}
        validated: List[Tuple[str, int, ParameterExpression]] = []
        for gate, qubit, angle in rotations:
            if gate not in allowed:
                raise ValueError(f"Unsupported pre/post rotation gate: {gate!r}. Allowed: {allowed}.")
            if not (0 <= qubit < n):
                raise ValueError(f"Rotation qubit {qubit} out of range for n={n}.")
            validated.append((gate, qubit, angle))
        return validated

    pre_rotations = _validate_rotations(pre_rotations)
    post_rotations = _validate_rotations(post_rotations)

    for rep in range(int(reps)):
        # Pre‑rotations
        for gate, qubit, angle in pre_rotations:
            getattr(qc, gate)(angle, qubit)
        if insert_barriers:
            qc.barrier()

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x_scaled[i]), i)

        # 2‑body entanglers
        for (i, j) in pairs:
            angle_2 = 2 * map2(x_scaled[i], x_scaled[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # 3‑body entanglers (if requested)
        for (i, j, k) in triples:
            angle_3 = 2 * map3(x_scaled[i], x_scaled[j], x_scaled[k])
            qc.append(CCXGate(), [i, j, k])
            qc.p(angle_3, k)
            qc.append(CCXGate(), [i, j, k])

        # Post‑rotations
        for gate, qubit, angle in post_rotations:
            getattr(qc, gate)(angle, qubit)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# OO wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapExtended(QuantumCircuit):
    """Object‑oriented wrapper for :func:`zz_feature_map_extended`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        pre_rotations: Sequence[Tuple[str, int, ParameterExpression]] | None = None,
        post_rotations: Sequence[Tuple[str, int, ParameterExpression]] | None = None,
        normalisation: float | None = None,
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
            pre_rotations=pre_rotations,
            post_rotations=post_rotations,
            normalisation=normalisation,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
