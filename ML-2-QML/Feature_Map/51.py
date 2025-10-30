"""Module providing a controlled‑modified RZZ‑entangler feature map for Qiskit.

This circuit extends the original ZZFeatureMapRZZ by:
- Using a shared pair‑scale that is independent of the data.
- Adding an optional global normalisation parameter.
- Introducing a new "cyclic" entanglement mode.
- Maintaining compatibility with Qiskit's data‑encoding workflows.

The module exposes both a functional helper `zz_feature_map_rzz_controlled` and
a class‑style wrapper `ZZFeatureMapRZZControlled`.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    n: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of qubit pairs for entanglement.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest‑neighbour pairs (0,1), (1,2),...
      - "circular" or "cyclic": linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of tuples [(i,j),...]
      - callable: f(n) -> sequence of (i,j)

    Raises:
        ValueError: if spec is invalid or pairs are out of range.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if entanglement in {"circular", "cyclic"}:
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(n))
        return [(int(i), int(j)) for (i, j) in pairs]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={n}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default mapping for single‑qubit rotations."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default mapping for two‑qubit entanglers."""
    return (pi - x) * (pi - y)

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
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a controlled‑modified RZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the entanglement block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. See :func:`_resolve_entanglement`.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional user‑supplied mapping from raw data to rotation angles.
        If None, defaults to linear mapping for single‑qubit rotations
        and a quadratic mapping for two‑qubit entanglers.
    parameter_prefix : str, default "x"
        Prefix for the input parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for visual clarity.
    pair_scale : float, default 1.0
        Global scaling factor applied to all RZZ angles. Must be non‑negative.
    normalise_data : bool, default False
        If True, the circuit includes a global scaling parameter ``norm`` that
        multiplies all input data. The user can set this to ``1/||x||`` to
        normalise the feature vector before binding.
    name : str | None, default None
        Name of the resulting circuit. If None, defaults to
        ``"ZZFeatureMapRZZControlled"``.

    Returns
    -------
    QuantumCircuit
        A parameterised feature‑map circuit.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``reps`` <= 0, or ``pair_scale`` < 0.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    if pair_scale < 0:
        raise ValueError("pair_scale must be non‑negative.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Input parameters
    x = ParameterVector(parameter_prefix, n)

    # Optional global normalisation parameter
    if normalise_data:
        norm = ParameterVector("norm", 1)
        scaled_x = [x[i] * norm[0] for i in range(n)]
    else:
        scaled_x = list(x)

    # Map functions
    if data_map_func is None:
        map1 = lambda xi: xi
        map2 = lambda xi, xj: (pi - xi) * (pi - xj)
    else:
        map1 = lambda xi: data_map_func([xi])
        map2 = lambda xi, xj: data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(scaled_x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            # RZZ angle uses shared pair_scale and is independent of data
            qc.rzz(2 * pair_scale, i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Store input parameters for binding
    if normalise_data:
        qc.input_params = list(x) + list(norm)  # type: ignore[attr-defined]
    else:
        qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified RZZ‑entangler feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the entanglement block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Optional mapping from raw data to rotation angles.
    parameter_prefix : str, default "x"
        Prefix for the input parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    pair_scale : float, default 1.0
        Global scaling factor for RZZ angles.
    normalise_data : bool, default False
        Include a global normalisation parameter.
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
        normalise_data: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension, reps, entanglement, data_map_func,
            parameter_prefix, insert_barriers, pair_scale, normalise_data, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
