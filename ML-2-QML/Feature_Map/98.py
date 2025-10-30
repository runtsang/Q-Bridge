"""Extended ZZFeatureMap with higher‑order interactions and optional rotations.

This module defines a parameterised quantum circuit that generalises the canonical
ZZFeatureMap by allowing:
  • optional triple‑qubit ZZ interactions (controlled‑phase via two CNOTs)
  • pre‑rotation around Y (optional)
  • post‑rotation around Z (optional)
  • simple normalisation of raw data to the range [0, π]
  • custom data‑mapping functions for arbitrary preprocessing

The design keeps the original CX–P–CX structure for pairwise ZZ gates and
adds the new features in a modular, backwards‑compatible way.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
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
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ₃(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Extended ZZFeatureMap builder
# ---------------------------------------------------------------------------

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    pre_rotation: bool = False,
    post_rotation: bool = False,
    normalize: bool = True,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map with optional higher‑order interactions and rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input feature vector.
    reps : int, default 2
        Number of repetitions of the base pattern.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of pairwise entanglement pairs.
    interaction_order : int, default 2
        2 for standard pairwise ZZ; 3 to add triple‑qubit ZZ interactions.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping from raw data to rotation angles. If ``None``, defaults
        to the canonical φ₁, φ₂, φ₃ functions.
    parameter_prefix : str, default "x"
        Prefix for the automatically generated ParameterVector.
    pre_rotation : bool, default False
        If ``True``, apply a Y‑rotation (R<sub>y</sub>) before the Hadamard
        preparation, using the same data‑dependent phases.
    post_rotation : bool, default False
        If ``True``, apply a Z‑rotation (R<sub>z</sub>) after all entanglers.
    normalize : bool, default True
        If ``True``, multiply all raw data by ``π`` to map them into
        the range [0, π]. This is a lightweight normalisation suitable for
        most supervised learning tasks.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for easier visualisation.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with a feature vector.

    Raises
    ------
    ValueError
        If input parameters are inconsistent (e.g. insufficient qubits for
        triple interactions, unsupported interaction order, etc.).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for pairwise interactions.")
    if interaction_order == 3 and feature_dimension < 3:
        raise ValueError("feature_dimension must be >= 3 for triple‑qubit interactions.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Parameter vector for raw data
    raw_params = ParameterVector(parameter_prefix, n)

    # Optional normalisation
    if normalize:
        raw_params = [pi * p for p in raw_params]

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
        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.ry(2 * map1(raw_params[i]), i)
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(raw_params[i]), i)
        if insert_barriers:
            qc.barrier()
        # Pairwise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(raw_params[i], raw_params[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        # Optional triple‑qubit ZZ
        if interaction_order == 3:
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        angle_3 = 2 * map3(raw_params[i], raw_params[j], raw_params[k])
                        # 3‑qubit controlled phase via 3 CNOTs
                        qc.cx(i, k)
                        qc.cx(j, k)
                        qc.p(angle_3, k)
                        qc.cx(j, k)
                        qc.cx(i, k)
        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.rz(2 * map1(raw_params[i]), i)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapExtended(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature‑map.

    The constructor builds the circuit via :func:`zz_feature_map_extended` and
    composes it into the subclass instance, exposing the same ``input_params``
    attribute for easy parameter binding.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input feature vector.
    reps : int, default 2
        Number of repetitions of the base pattern.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of pairwise entanglement pairs.
    interaction_order : int, default 2
        2 for standard pairwise ZZ; 3 to add triple‑qubit ZZ interactions.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping from raw data to rotation angles.
    parameter_prefix : str, default "x"
        Prefix for the automatically generated ParameterVector.
    pre_rotation : bool, default False
        If ``True``, apply a Y‑rotation before the Hadamard preparation.
    post_rotation : bool, default False
        If ``True``, apply a Z‑rotation after all entanglers.
    normalize : bool, default True
        Normalise raw data into the range [0, π] by scaling.
    insert_barriers : bool, default False
        Insert barriers between logical blocks.
    name : str, default "ZZFeatureMapExtended"
        Optional circuit name.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        pre_rotation: bool = False,
        post_rotation: bool = False,
        normalize: bool = True,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalize=normalize,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
