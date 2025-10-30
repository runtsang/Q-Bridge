"""Symmetric ZZ‑Feature Map with shared pair parameters and optional global rotation.

This module implements a controlled‑modification of the canonical ZZFeatureMap.
Key changes:
- A single shared parameter controls all pair‑wise ZZ interactions.
- Optional global RY rotation applied after each repetition.
- Parameter vectors are exposed as `input_params`, `pair_params`, and `global_params`.
- The circuit remains compatible with Qiskit data‑encoding workflows.
"""

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
# Symmetric ZZ Feature Map (controlled modification)
# ---------------------------------------------------------------------------

def zz_symmetric_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    pair_parameter_prefix: str = "y",
    insert_barriers: bool = False,
    global_rotation: bool = False,
    global_rotation_prefix: str = "g",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetric ZZ‑feature‑map with shared pair parameters.

    The circuit follows the canonical pattern:
        H → P(2·φ1) on each qubit → ZZ entanglers via CX–P–CX
    but with the following controlled modifications:
        * All pair‑wise ZZ interactions use a single shared parameter.
        * An optional global RY rotation can be applied after each repetition.
    Parameters
    ----------
    feature_dimension : int
        Number of qubits / input features. Must be ≥ 2.
    reps : int, default 2
        Number of repetitions (depth).
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of qubit pairs to entangle.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping of raw feature parameters to rotation angles.
        If ``None``, defaults to the canonical φ1 and φ2.
    parameter_prefix : str, default "x"
        Prefix for single‑qubit rotation parameters.
    pair_parameter_prefix : str, default "y"
        Prefix for the shared pair‑wise rotation parameter.
    insert_barriers : bool, default False
        Insert barriers between circuit sections for clarity.
    global_rotation : bool, default False
        If ``True``, a global RY rotation with a shared parameter is applied
        after each repetition.
    global_rotation_prefix : str, default "g"
        Prefix for the global rotation parameter.
    name : str | None, default None
        Optional name for the QuantumCircuit.
    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZSymmetricFeatureMap")

    # Single‑qubit parameters
    single_params = ParameterVector(parameter_prefix, n)

    # Shared pair‑wise parameter
    pair_params = ParameterVector(pair_parameter_prefix, 1)

    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    if global_rotation:
        global_params = ParameterVector(global_rotation_prefix, 1)
    else:
        global_params = None

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(single_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglement with shared parameter
        for (i, j) in pairs:
            angle = 2 * pair_params[0] * map2(single_params[i], single_params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if global_rotation:
            for i in range(n):
                qc.ry(2 * global_params[0], i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Expose parameters for external binding
    qc.input_params = single_params  # type: ignore[attr-defined]
    qc.pair_params = pair_params     # type: ignore[attr-defined]
    if global_rotation:
        qc.global_params = global_params  # type: ignore[attr-defined]

    return qc


class ZZSymmetricFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the symmetric ZZ‑feature‑map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        pair_parameter_prefix: str = "y",
        insert_barriers: bool = False,
        global_rotation: bool = False,
        global_rotation_prefix: str = "g",
        name: str = "ZZSymmetricFeatureMap",
    ) -> None:
        built = zz_symmetric_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            pair_parameter_prefix=pair_parameter_prefix,
            insert_barriers=insert_barriers,
            global_rotation=global_rotation,
            global_rotation_prefix=global_rotation_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.pair_params = built.pair_params  # type: ignore[attr-defined]
        if global_rotation:
            self.global_params = built.global_params  # type: ignore[attr-defined]


__all__ = ["ZZSymmetricFeatureMap", "zz_symmetric_feature_map"]
