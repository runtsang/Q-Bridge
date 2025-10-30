"""Extended ZZ‑feature map with optional pre/post rotations, higher‑order interactions, and data normalisation.

The design builds on the canonical ZZ‑feature map (Hadamard + P + CX–P–CX for ZZ) and introduces:
  • **pre_rotation** and **post_rotation** – optional Rz rotations applied before the Hadamards or after the entanglers.
  • **interaction_order** – default 2 (pairwise), can be set to 3 to add triple‑qubit ZZ interactions (CX–CX–Rz–CX–CX).
  • **normalize** – if True, each classical feature is scaled to [0, π] before mapping.
  • Custom data mapping functions for pairwise (`data_map_func`) and triple (`data_map_func_3`) interactions.
  • Flexible entanglement patterns: "full", "linear", "circular", explicit list, or callable.

Both a functional helper (`extended_zz_feature_map`) and a subclass (`ExtendedZZFeatureMap`) are provided for convenience.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

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
      - "circular": linear + wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) → sequence of (i, j)
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


def _resolve_triplets(
    num_qubits: int,
    triplet_spec: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Return a list of three‑qubit triplets according to a simple spec.

    Supported specs:
      - "full": all combinations i < j < k
      - "linear": consecutive triplets (0,1,2), (1,2,3), …
    """
    if isinstance(triplet_spec, str):
        if triplet_spec == "full":
            return [(i, j, k) for i in range(num_qubits)
                    for j in range(i + 1, num_qubits)
                    for k in range(j + 1, num_qubits)]
        if triplet_spec == "linear":
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        raise ValueError(f"Unknown triplet spec string: {triplet_spec!r}")

    if callable(triplet_spec):
        triplets = list(triplet_spec(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in triplets]

    triplets = [(int(i), int(j), int(k)) for (i, j, k) in triplet_spec]  # type: ignore[arg-type]
    for (i, j, k) in triplets:
        if len({i, j, k})!= 3:
            raise ValueError("Triplet indices must be distinct.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits and 0 <= k < num_qubits):
            raise ValueError(f"Triplet {(i, j, k)} out of range for n={num_qubits}.")
    return triplets


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
# Extended ZZFeatureMap (functional builder)
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    triplet_spec: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "linear",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    data_map_func_3: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    interaction_order: int = 2,
    parameter_prefix: str = "x",
    normalize: bool = True,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2 for pairwise interactions,
        >= 3 if `interaction_order` > 2.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of pairwise entanglement pairs.
    triplet_spec : str | Sequence[Tuple[int, int, int]] | Callable
        Specification of triple‑qubit triplets when `interaction_order` >= 3.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Function mapping a list of parameters to a single parameter expression.
        If None, defaults to a simple mapping that optionally normalises data.
    data_map_func_3 : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Mapping function for triple interactions. If None, defaults to a simple mapping.
    interaction_order : int, default 2
        2 for pairwise ZZ, 3 for adding triple‑qubit ZZ interactions.
    parameter_prefix : str, default "x"
        Prefix for the generated ParameterVector.
    normalize : bool, default True
        If True, each feature is scaled to [0, π] before mapping.
    pre_rotation : bool, default False
        If True, apply Rz(θ) on each qubit before the Hadamard at each repetition.
    post_rotation : bool, default False
        If True, apply Rz(θ) on each qubit after the entanglers at each repetition.
    insert_barriers : bool, default False
        Insert barriers between logical sections for clarity.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZ features.")
    if interaction_order == 3 and feature_dimension < 3:
        raise ValueError("feature_dimension must be >= 3 for triple‑qubit interactions (interaction_order=3).")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Helper scaling function
    def _scale(param: ParameterExpression) -> ParameterExpression:
        return pi * param if normalize else param

    # Default mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return _scale(xi)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

    if data_map_func_3 is None:
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return _scale(xi) * _scale(xj) * _scale(xk)
    else:
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func_3([xi, xj, xk])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Resolve triplets if needed
    triplets: List[Tuple[int, int, int]] = []
    if interaction_order == 3:
        triplets = _resolve_triplets(n, triplet_spec)

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.rz(_scale(x[i]), i)
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map1(x[i]) * map1(x[j])  # using map1 for simplicity
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Triple‑qubit ZZ via CX–CX–Rz–CX–CX (if enabled)
        if interaction_order == 3:
            for (i, j, k) in triplets:
                angle_3 = 2 * map3(x[i], x[j], x[k])  # factor 2 for consistency
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle_3, k)
                qc.cx(j, k)
                qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.rz(_scale(x[i]), i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Extended ZZFeatureMap (class wrapper)
# ---------------------------------------------------------------------------

class ExtendedZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature map.

    Instantiation behaves like the functional builder, but the resulting
    object is a concrete ``QuantumCircuit`` that can be composed with other
    circuits or used directly in a Qiskit workflow.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        triplet_spec: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "linear",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        data_map_func_3: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        interaction_order: int = 2,
        parameter_prefix: str = "x",
        normalize: bool = True,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        insert_barriers: bool = False,
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            triplet_spec=triplet_spec,
            data_map_func=data_map_func,
            data_map_func_3=data_map_func_3,
            interaction_order=interaction_order,
            parameter_prefix=parameter_prefix,
            normalize=normalize,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
