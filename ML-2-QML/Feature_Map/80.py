"""Controlled‑modification ZZ‑feature map with shared pair angles and optional data normalisation."""
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
      - "circular": linear plus wrap‑around
      - explicit list of pairs
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


def _default_map_2_shared(
    x: ParameterExpression,
    y: ParameterExpression,
    shared: ParameterExpression | float,
) -> ParameterExpression:
    """Default φ2(x, y) = shared * (π - x)(π - y)."""
    return shared * (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Controlled‑Modification ZZFeatureMap
# ---------------------------------------------------------------------------

def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    shared_pair_angle: bool = False,
    data_normalisation: bool = False,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a controlled‑modification ZZ‑feature‑map.

    The circuit is similar to the canonical ZZ‑feature‑map but introduces two
    optional structural changes:

    * **shared_pair_angle** – when True, a single shared parameter controls all
      two‑qubit ZZ entanglers. This reduces the parameter count and forces a
      symmetric interaction pattern.
    * **data_normalisation** – when True, the user is expected to normalise each
      classical feature to the interval [0, π] before binding; no runtime
      scaling is performed.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / input features; must be ≥ 2.
    reps : int, default 2
        Depth of the feature map.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    data_map_func : Callable, optional
        Custom mapping from raw parameters to the desired functional form.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    shared_pair_angle : bool, default False
        If True, all pair interactions share one angle parameter.
    data_normalisation : bool, default False
        If True, normalise each feature to [0, π] before mapping.
    insert_barriers : bool, default False
        Insert barriers for visual clarity.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapControlled.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Base parameter vector for data mapping
    x = ParameterVector(parameter_prefix, n)

    # Optional shared parameter for pair interactions
    shared_angle = ParameterVector(f"{parameter_prefix}_shared", 1) if shared_pair_angle else None

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Choose mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        if shared_pair_angle:
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return _default_map_2_shared(xi, xj, shared_angle[0])
        else:
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return _default_map_2_shared(xi, xj, 1.0)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        if shared_pair_angle:
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return _default_map_2_shared(xi, xj, shared_angle[0])
        else:
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return data_map_func([xi, xj])

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            angle1 = 2 * map1(x[i])
            qc.p(angle1, i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ entanglers
        for (i, j) in pairs:
            angle2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach parameter vector for binding
    qc.input_params = x  # type: ignore[attr-defined]
    if shared_pair_angle:
        qc.shared_params = shared_angle  # type: ignore[attr-defined]

    return qc


class ZZFeatureMapControlled(QuantumCircuit):
    """Object‑oriented wrapper for the controlled‑modification ZZFeatureMap."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        shared_pair_angle: bool = False,
        data_normalisation: bool = False,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            shared_pair_angle=shared_pair_angle,
            data_normalisation=data_normalisation,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        if shared_pair_angle:
            self.shared_params = built.shared_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled"]
