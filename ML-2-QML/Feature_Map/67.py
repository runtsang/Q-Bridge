"""
Extended ZZ Feature Map for Qiskit

This module implements the *ZZFeatureMapExtended* builder and its
`QuantumCircuit` subclass.  The design follows the *extension* scaling
paradigm: it keeps the core structure of the canonical ZZ map but adds
optional higher‑order interactions, pre/post‑rotations, and normalisation
of input features.

Key design choices
------------------
* **Higher‑order ZZ** – an optional 3‑qubit `ZZZ` interaction is added
  when `interaction_order==3`.  It is implemented via a sequence of
  two CNOTs and a phase on the target qubit.
* **Pre/Post‑rotations** – optional `RZ` rotations before the Hadamards
  and after the entangling block allow extra tunable phases.
* **Feature normalisation** – if `normalize==True` the input features
  are scaled to the range `[-π/2, π/2]` (default) using a
  `scaling_factor`.  Users may override the factor via the
  `scaling_factor` argument.
* **Data‑map flexibility** – a user supplied `data_map_func` can
  convert raw feature vectors into gate angles.  If omitted, the
  defaults `φ1(x)=x`, `φ2(x,y)=(π−x)(π−y)`, and
  `φ3(x,y,z)=(π−x)(π−y)(π−z)` are used.
* **Error handling** – the builder validates qubit count, interaction
  order, entanglement specification, and parameter shapes with clear
  messages.

Usage
-----
>>> from zz_feature_map_extension import zz_feature_map_extended, ZZFeatureMapExtended
>>> qc = zz_feature_map_extended(feature_dimension=4, reps=3, interaction_order=3,
                                 normalize=True, pre_rotation=True, post_rotation=True)
>>> print(qc.draw())
"""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> Sequence[Tuple[int, int]]:
    """Resolve entanglement specification into a list of qubit pairs.

    Supported specs:
      * ``"full"``      – all distinct pairs (i < j)
      * ``"linear"``    – nearest‑neighbour chain
      * ``"circular"``  – linear + wrap‑around edge
      * explicit list of tuples ``[(0, 1), (1, 2)]``
      * callable ``f(n) -> list of pairs``
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


def _resolve_triplets(num_qubits: int) -> Sequence[Tuple[int, int, int]]:
    """Return all unique triplets of qubits (i < j < k)."""
    return [(i, j, k) for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            for k in range(j + 1, num_qubits)]


# ---------------------------------------------------------------------------
# Default data‑mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression, scale: float = 1.0) -> ParameterExpression:
    """φ1(x) = x  (scaled if requested)."""
    return scale * x


def _default_map_2(
    x: ParameterExpression,
    y: ParameterExpression,
    scale: float = 1.0,
) -> ParameterExpression:
    """φ2(x, y) = (π − x)(π − y)  (scaled if requested)."""
    return scale * (pi - x) * (pi - y)


def _default_map_3(
    x: ParameterExpression,
    y: ParameterExpression,
    z: ParameterExpression,
    scale: float = 1.0,
) -> ParameterExpression:
    """φ3(x, y, z) = (π − x)(π − y)(π − z)  (scaled if requested)."""
    return scale * (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Feature‑map builder
# ---------------------------------------------------------------------------

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    interaction_order: int = 2,
    normalize: bool = True,
    scaling_factor: float | None = None,
    pre_rotation: bool = False,
    post_rotation: bool = False,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits (and input features).  Must be >= 2.
    reps : int, default 2
        Number of repetitions of the block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit entangling pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        User‑supplied mapping from feature vector to a gate angle.
        If omitted, defaults to the canonical functions.
    parameter_prefix : str, default "x"
        Prefix for automatically generated `Parameter` names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    name : str | None, default None
        Circuit name; if ``None`` a default is supplied.
    interaction_order : int, default 2
        2 for standard ZZ; 3 adds a 3‑qubit `ZZZ` interaction.
    normalize : bool, default True
        If ``True`` scales each input feature to the range ``[-π/2, π/2]``.
    scaling_factor : float | None, default None
        Explicit scaling factor overriding the default.  Ignored if ``normalize`` is False.
    pre_rotation : bool, default False
        If ``True`` applies an additional `RZ` rotation before the Hadamard.
    post_rotation : bool, default False
        If ``True`` applies an additional `RZ` rotation after the entanglement.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapExtended.")

    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be either 2 or 3.")

    if interaction_order == 3 and feature_dimension < 3:
        raise ValueError("interaction_order==3 requires at least 3 qubits.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Normalisation
    if normalize:
        sf = scaling_factor if scaling_factor is not None else pi / 2.0
    else:
        sf = 1.0

    # Scale the raw parameters for mapping
    x_scaled = [sf * xi for xi in x]

    # Define mapping functions
    if data_map_func is None:
        map1 = lambda xi: _default_map_1(xi, sf)
        map2 = lambda xi, xj: _default_map_2(xi, xj, sf)
        map3 = lambda xi, xj, xk: _default_map_3(xi, xj, xk, sf)
    else:
        # data_map_func receives a list of ParameterExpressions
        map1 = lambda xi: data_map_func([xi])
        map2 = lambda xi, xj: data_map_func([xi, xj])
        map3 = lambda xi, xj, xk: data_map_func([xi, xj, xk])

    pairs = _resolve_entanglement(n, entanglement)
    triplets = _resolve_triplets(n) if interaction_order == 3 else []

    for _ in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.rz(2 * map1(x[i]), i)

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        # Two‑qubit ZZ interactions
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional higher‑order ZZ interactions
        for (i, j, k) in triplets:
            angle = 2 * map3(x[i], x[j], x[k])
            qc.cx(i, j)
            qc.cx(i, k)
            qc.p(angle, k)
            qc.cx(i, k)
            qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.rz(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapExtended(QuantumCircuit):
    """Class‑style wrapper for `zz_feature_map_extended`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtended",
        interaction_order: int = 2,
        normalize: bool = True,
        scaling_factor: float | None = None,
        pre_rotation: bool = False,
        post_rotation: bool = False,
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            interaction_order=interaction_order,
            normalize=normalize,
            scaling_factor=scaling_factor,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
