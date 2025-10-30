"""ZZFeatureMapRZZControlled variant with symmetric entanglement, shared scaling and normalisation."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``full``   : all‑to‑all pairs (i < j)
      - ``linear`` : nearest neighbors (0,1), (1,2), …
      - ``circular`` : linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs, e.g. ``[(0, 2), (1, 3)]``
      - callable ``f(num_qubits) -> sequence of (i, j)``

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair is invalid.
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


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# --------------------------------------------------------------------------- #
# Functional feature map
# --------------------------------------------------------------------------- #

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    global_pair_scale: float = 1.0,
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    RZZ‑based feature map with controlled modifications:

    * Symmetric entanglement (unchanged from the seed).
    * Optional global scaling of all two‑qubit angles per repetition.
    * Optional normalisation of data to the [0, π] range.
    * Shared scaling factor for all pair entanglers.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default=2
        Number of repetitions of the base layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Function mapping feature(s) to an angle. If None, defaults to
        φ1(x) = x, φ2(x, y) = (π − x)(π − y).
    parameter_prefix : str, default="x"
        Prefix for automatically generated parameter names.
    insert_barriers : bool, default=False
        Insert barriers between layers for visual clarity.
    pair_scale : float, default=1.0
        Local scaling of individual pair angles.
    global_pair_scale : float, default=1.0
        Global scaling applied to all pair angles within a repetition.
    normalise_data : bool, default=False
        If True, automatically scale each input feature from [0, 1] to [0, π].
    name : str | None, default=None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If input parameters are out of bounds.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_scale < 0 or global_pair_scale < 0:
        raise ValueError("pair_scale and global_pair_scale must be non‑negative.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for classical features
    x = ParameterVector(parameter_prefix, n)

    # Define data mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit rotations
        for i in range(n):
            phi1 = map1(x[i])
            if normalise_data:
                phi1 = pi * phi1
            qc.p(2 * phi1, i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit entanglers
        for (i, j) in pairs:
            phi2 = map2(x[i], x[j])
            if normalise_data:
                phi2 = pi * phi2
            angle = 2 * pair_scale * global_pair_scale * phi2
            qc.rzz(angle, i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class‑style wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapRZZControlled(QuantumCircuit):
    """
    Class‑style wrapper for the RZZ‑controlled feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default=2
        Number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Data‑to‑angle mapping function.
    parameter_prefix : str, default="x"
        Prefix for parameters.
    insert_barriers : bool, default=False
        Insert barriers.
    pair_scale : float, default=1.0
        Local pair scaling.
    global_pair_scale : float, default=1.0
        Global pair scaling per repetition.
    normalise_data : bool, default=False
        Normalise input features to [0, π].
    name : str, default="ZZFeatureMapRZZControlled"
        Circuit name.

    Attributes
    ----------
    input_params : ParameterVector
        Reference to the circuit’s parameter vector for binding.
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
        global_pair_scale: float = 1.0,
        normalise_data: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            global_pair_scale,
            normalise_data,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
