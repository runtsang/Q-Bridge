"""ZZFeatureMapRZZExtension – a scalable, Qiskit‑compatible feature map.

The module defines two entry points:
  * `zz_feature_map_rzz_extension` – functional interface.
  * `ZZFeatureMapRZZExtension` – subclass of `QuantumCircuit` for OO usage.
The code‑base mirrors the original seed but further sparsifies and elevates the entanglement depth.
"""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ----------------------------------------------------------------------
# Utility: pairwise entanglement resolution
# ----------------------------------------------------------------------
def _resolve_entanglement(
    *,
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) -> sequence of (i, j)

    Raises
    ------
    ValueError
        If an unsupported spec is provided or a pair is invalid.
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


# ----------------------------------------------------------------------
# Default mapping functions
# ----------------------------------------------------------------------
def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ----------------------------------------------------------------------
# Functional interface
# ----------------------------------------------------------------------
def zz_feature_map_rzz_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    name: str | None = None,
    use_pre_rotation: bool = False,
    pre_rotation_angle: float = pi / 2,
    normalisation_factor: float = 1.0,
    interaction_order: int = 2,
) -> QuantumCircuit:
    """
    Construct a scalable ZZ feature map using native ``rzz`` gates.

    This variant extends the original RZZ entangler with:
    - optional pre‑rotation on each qubit (toggle via ``use_pre_rotation``)
    - a global normalisation factor that scales all rotation angles
    - optional higher‑order interaction layers (currently 3‑body) controlled by ``interaction_order``

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default 2
        Number of feature‑map repetitions. Must be >= 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of two‑qubit entanglement pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Function to map a list of parameters to an angle. If ``None``, default
        polynomial maps are used.
    parameter_prefix : str, default "x"
        Prefix for the generated `ParameterVector`.
    insert_barriers : bool, default False
        Insert barriers between layers for visual clarity.
    pair_scale : float, default 1.0
        Scaling factor applied to pairwise interaction angles.
    name : str | None, default None
        Optional circuit name; defaults to ``"ZZFeatureMapRZZExtension"``.
    use_pre_rotation : bool, default False
        If ``True``, apply a pre‑rotation on each qubit before the H layer.
    pre_rotation_angle : float, default π/2
        Angle multiplier for the pre‑rotation.
    normalisation_factor : float, default 1.0
        Global factor scaling all rotation angles.
    interaction_order : int, default 2
        If >2, an additional higher‑order interaction layer is added.
        Currently supports 3‑body interactions.

    Returns
    -------
    QuantumCircuit
        A parameterised feature‑map circuit with attributes:
        * ``input_params`` – the `ParameterVector` used for data binding.
    """
    # Validation
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_scale < 0:
        raise ValueError("pair_scale must be non‑negative.")
    if normalisation_factor < 0:
        raise ValueError("normalisation_factor must be non‑negative.")
    if interaction_order < 2:
        raise ValueError("interaction_order must be >= 2.")
    n = int(feature_dimension)

    if interaction_order > n:
        raise ValueError("interaction_order cannot exceed the number of qubits.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtension")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Mapping functions
    map1 = (_default_map_1 if data_map_func is None else
            lambda xi: data_map_func([xi]))
    map2 = (_default_map_2 if data_map_func is None else
            lambda xi, xj: data_map_func([xi, xj]))
    map3 = (_default_map_3 if data_map_func is None else
            lambda xi, xj, xk: data_map_func([xi, xj, xk]))

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(num_qubits=n, entanglement=entanglement)

    # Higher‑order triples (currently 3‑body)
    triples: List[Tuple[int, int, int]] = []
    if interaction_order == 3:
        triples = list(itertools.combinations(range(n), 3))
    elif interaction_order > 3:
        raise NotImplementedError("Interaction orders >3 are not supported yet.")

    for rep in range(reps):
        # Optional pre‑rotation
        if use_pre_rotation:
            for i in range(n):
                qc.rz(normalisation_factor * pre_rotation_angle * map1(x[i]), i)

        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * normalisation_factor * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise entanglement
        for (i, j) in pairs:
            qc.rzz(2 * normalisation_factor * pair_scale * map2(x[i], x[j]), i, j)

        # Higher‑order interaction layer
        if triples:
            for (i, j, k) in triples:
                # Use a single RZZ gate between i and k with angle incorporating all three features.
                qc.rzz(2 * normalisation_factor * pair_scale * map3(x[i], x[j], x[k]), i, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ----------------------------------------------------------------------
# Class‑style wrapper
# ----------------------------------------------------------------------
class ZZFeatureMapRZZExtension(QuantumCircuit):
    """Class‑style wrapper for the scalable RZZ‑entangled feature map.

    Parameters are identical to :func:`zz_feature_map_rzz_extension`.  The
    constructor builds the underlying circuit and composes it into the
    current instance.  The resulting object exposes the ``input_params``
    attribute for data binding.

    Example
    -------
    >>> from zz_feature_map_rzz_extension import ZZFeatureMapRZZExtension
    >>> circ = ZZFeatureMapRZZExtension(feature_dimension=4, reps=3)
    >>> circ.draw()
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
        name: str = "ZZFeatureMapRZZExtension",
        use_pre_rotation: bool = False,
        pre_rotation_angle: float = pi / 2,
        normalisation_factor: float = 1.0,
        interaction_order: int = 2,
    ) -> None:
        built = zz_feature_map_rzz_extension(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            name,
            use_pre_rotation,
            pre_rotation_angle,
            normalisation_factor,
            interaction_order,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtension", "zz_feature_map_rzz_extension"]
