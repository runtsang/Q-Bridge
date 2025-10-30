"""ZZFeatureMapControlled – controlled-modification variant of the canonical ZZ feature map.

This module provides a function and a QuantumCircuit subclass that build a
feature‑map circuit with additional structural knobs:

* Mirror entanglement pattern (i ↔ n‑1‑i) alongside the standard patterns.
* Global scaling of all rotation angles (`scaling_factor`).
* Shared phase applied to every qubit (`shared_phase`).
* Global entanglement strength multiplier (`shared_entanglement_strength`).
* Optional pre‑rotation before the Hadamard layer (`include_pre_rotations`,
  `pre_rotation_angle`).
* Optional normalization of inputs to the [0, π] range (`normalize`).

The core construction remains the CX–P–CX ZZ entanglement, but the added
parameters give users fine‑grained control over depth, expressivity, and
parameter sharing.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Entanglement resolution
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2),...
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
      - "mirror": (0,n‑1), (1,n‑2), … (midpoint not entangled for odd n)
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
        if entanglement == "mirror":
            pairs = []
            for i in range(num_qubits // 2):
                pairs.append((i, num_qubits - 1 - i))
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


# ---------------------------------------------------------------------------
# Default data mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Feature‑map builder (controlled‑modification variant)
# ---------------------------------------------------------------------------

def zz_feature_map_controlled_modification(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    scaling_factor: float = 1.0,
    shared_phase: float = 0.0,
    shared_entanglement_strength: float = 1.0,
    include_pre_rotations: bool = False,
    pre_rotation_angle: float = pi / 4,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a controlled‑modification variant of the canonical ZZ feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits (must be >= 1).
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. Supported strings: "full", "linear",
        "circular", "mirror".  Custom lists or callables are also accepted.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Optional user‑supplied mapping that receives the list of
        classical parameters for a given term and returns a rotation
        angle.  If omitted, the default mappings are used.
    parameter_prefix : str, default "x"
        Prefix for the automatically generated ParameterVector.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    scaling_factor : float, default 1.0
        Global multiplier applied to all rotation angles.
    shared_phase : float, default 0.0
        A constant phase added to each qubit after the Hadamard layer.
    shared_entanglement_strength : float, default 1.0
        Multiplier applied to all ZZ interaction angles.
    include_pre_rotations : bool, default False
        If True, a fixed Rz rotation is applied to each qubit before the
        Hadamard preparation.
    pre_rotation_angle : float, default π/4
        Angle for the optional pre‑rotation.
    normalize : bool, default False
        If True, the raw classical features are scaled to [0, π] before
        being passed to the default mapping functions.
    name : str | None, default None
        Circuit name.  If None, a default is used.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit has an
        ``input_params`` attribute containing the ParameterVector.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 1 or if the entanglement specification
        contains invalid pairs.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1 for ZZFeatureMapControlled.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Parameter vector for the classical data
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Determine mapping functions
    if data_map_func is None:
        if normalize:
            def map1(xi: ParameterExpression) -> ParameterExpression:
                return pi * xi
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return pi * (xi + xj)
        else:
            map1 = _default_map_1
            map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    for rep in range(int(reps)):
        # Optional pre‑rotations
        if include_pre_rotations:
            for i in range(n):
                qc.rz(pre_rotation_angle, i)

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases (with shared phase)
        for i in range(n):
            angle1 = 2 * scaling_factor * map1(x[i])
            qc.p(angle1, i)
            if shared_phase!= 0.0:
                qc.p(shared_phase, i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglement via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * scaling_factor * map2(x[i], x[j]) * shared_entanglement_strength
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification ZZ feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits.
    reps : int, default 2
        Number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom data mapping function.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector.
    insert_barriers : bool, default False
        Whether to insert barriers.
    scaling_factor : float, default 1.0
        Global angle scaling.
    shared_phase : float, default 0.0
        Shared phase added to each qubit.
    shared_entanglement_strength : float, default 1.0
        Entanglement strength multiplier.
    include_pre_rotations : bool, default False
        Whether to include pre‑rotations.
    pre_rotation_angle : float, default π/4
        Angle for the pre‑rotation.
    normalize : bool, default False
        Whether to normalise inputs to [0, π].
    name : str, default "ZZFeatureMapControlled"
        Circuit name.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        scaling_factor: float = 1.0,
        shared_phase: float = 0.0,
        shared_entanglement_strength: float = 1.0,
        include_pre_rotations: bool = False,
        pre_rotation_angle: float = pi / 4,
        normalize: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled_modification(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            scaling_factor=scaling_factor,
            shared_phase=shared_phase,
            shared_entanglement_strength=shared_entanglement_strength,
            include_pre_rotations=include_pre_rotations,
            pre_rotation_angle=pre_rotation_angle,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled_modification"]
