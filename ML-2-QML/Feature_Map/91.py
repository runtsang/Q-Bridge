"""ExtendedZZFeatureMap builder (Hadamard + ZZ entanglement via CX–P–CX, optional 3‑body interactions and pre/post rotations)."""

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
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2),...
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

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _triples(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Generate all unique triples (i<j<k) for 3‑body interactions."""
    return [(i, j, k) for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            for k in range(j + 1, num_qubits)]


# ---------------------------------------------------------------------------
# Default data‑mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit data mapping φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit data mapping φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default three‑qubit data mapping φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Extended ZZ Feature Map
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotations: bool = False,
    post_rotations: bool = False,
    rotation_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build the Extended ZZ feature‑map circuit.

    The circuit extends the canonical ZZFeatureMap by optionally adding
    three‑body ZZ interactions and pre/post RX rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits.
    reps : int, default 2
        Number of repetitions of the block.
    entanglement : str | Sequence[Tuple[int,int]] | Callable
        Entanglement pattern.
    interaction_order : int, default 2
        Highest interaction order to include (2 or 3). 3 adds three‑body gates.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Custom mapping from data to rotation angles.  If None the default
        φ1, φ2, φ3 are used.
    parameter_prefix : str, default "x"
        Prefix for data parameters.
    insert_barriers : bool, default False
        Insert barriers between logical blocks.
    pre_rotations : bool, default False
        If True prepend an RX(θ_i) on each qubit before the H gate.
    post_rotations : bool, default False
        If True append an RX(θ_i) on each qubit after the final repetition.
    rotation_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZ interactions.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if interaction_order == 3 and feature_dimension < 3:
        raise ValueError("feature_dimension must be >= 3 to support 3‑body interactions.")
    if reps < 1:
        raise ValueError("reps must be a positive integer.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    # Data parameters
    x = ParameterVector(parameter_prefix, n)

    # Rotation parameters if requested
    theta = ParameterVector(rotation_prefix, n) if pre_rotations or post_rotations else None

    # Map functions
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

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        if pre_rotations:
            for i in range(n):
                qc.rx(theta[i], i)
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        if interaction_order == 3:
            for (i, j, k) in _triples(n):
                angle_3 = 2 * map3(x[i], x[j], x[k])
                qc.cx(i, j)
                qc.cx(i, k)
                qc.p(angle_3, k)
                qc.cx(i, k)
                qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()
    if post_rotations:
        for i in range(n):
            qc.rx(theta[i], i)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ExtendedZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the Extended ZZ feature map."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotations: bool = False,
        post_rotations: bool = False,
        rotation_prefix: str = "theta",
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            pre_rotations=pre_rotations,
            post_rotations=post_rotations,
            rotation_prefix=rotation_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
