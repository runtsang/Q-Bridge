"""ZZFeatureMapRZZControlled: symmetric pair‑coupling with shared parameters."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs.

    Supported specs:
      * ``"full"``   – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest‑neighbour pairs (0,1), (1,2), …
      * ``"circular"`` – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs like ``[(0, 2), (1, 3)]``
      * callable: ``f(num_qubits) -> sequence of (i, j)``
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


# --------------------------------------------------------------------------- #
# Feature‑map construction
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """Symmetric RZZ feature map with a shared pair‑coupling angle.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be at least 2.
    reps : int, default 2
        Number of repetitions of the feature‑map layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Function mapping a list of parameters to a single parameter expression.
        If None, defaults to the seed's φ1(x)=x and φ2(x,y)=(π−x)(π−y).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for readability.
    pair_scale : float, default 1.0
        Scaling factor applied to the shared pair‑coupling angle.
    name : str | None, default None
        Name of the circuit. If None, defaults to ``"ZZFeatureMapRZZControlled"``.

    Returns
    -------
    QuantumCircuit
        A parameterized quantum circuit implementing the symmetric RZZ feature map.

    Notes
    -----
    * The single pair‑coupling angle is computed as the sum of the pair‑map
      contributions over all entangled pairs, multiplied by ``pair_scale``.
    * Single‑qubit rotations use the map1 function applied to each qubit.
    * The circuit exposes ``input_params`` for easy parameter binding.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for classical features
    x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - xi) * (pi - xj)
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
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Shared pair‑coupling angle: sum over all pairs
        pair_angle = sum(map2(x[i], x[j]) for (i, j) in pairs)
        pair_angle = pair_scale * 2 * pair_angle  # factor 2 for RZZ convention

        for (i, j) in pairs:
            qc.rzz(pair_angle, i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the symmetric RZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be at least 2.
    reps : int, default 2
        Number of repetitions of the feature‑map layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Function mapping a list of parameters to a single parameter expression.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for readability.
    pair_scale : float, default 1.0
        Scaling factor applied to the shared pair‑coupling angle.
    name : str, default "ZZFeatureMapRZZControlled"
        Name of the circuit.
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
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
