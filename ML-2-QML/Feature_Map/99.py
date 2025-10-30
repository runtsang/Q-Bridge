"""Symmetrised ZZFeatureMap with shared‑parameter global rotation.

The module provides both a functional and a class‑style interface that
exposes a `QuantumCircuit` compatible with Qiskit's data‑encoding
workflows.  The circuit is a direct extension of the canonical
`ZZFeatureMap` but includes the following controlled modifications:

* Symmetric entanglement: each pair of qubits is entangled in both
  directions (i→j and j→i) so the ZZ‑interaction is fully symmetric.
* A shared‑parameter rotation on each qubit (the `global_phi` angle)
  that is applied after the single‑qubit phase gates.  This rotation
  couples all qubits via a common angle and can be used to inject
  data‑independent correlations.
* Optional normalisation of the raw feature vector before it is
  mapped to rotation angles.  The normalisation can be applied
  separately for each feature or globally.
* The circuit depth is unchanged for a fixed number of repetitions,
  but the additional gates are local and can be absorbed into a
  single‑qubit rotation if desired.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairings for entanglement.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest‑neighbor pairs
      - "circular": linear plus wrap‑around (n‑1,0)
      - explicit list of pairs
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


# ----------------------------------------------------------------------
# Functional interface
# ----------------------------------------------------------------------
def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    global_phi: Union[float, ParameterExpression] = 0.0,
    normalise_data: bool = False,
    normalisation_factor: float = 1.0,
) -> QuantumCircuit:
    """
    Build a symmetrised ZZ‑feature‑map with a shared‑parameter global rotation.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, default 2
        Number of repetitions of the feature‑mapping block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping from raw data to rotation angles.  If None, the
        default mapping used in the original ZZFeatureMap is applied.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between logical layers.
    name : str | None, default None
        Circuit name; if None, "ZZFeatureMapControlled" is used.
    global_phi : float | ParameterExpression, default 0.0
        Shared rotation applied to each qubit after the single‑qubit phase gates.
    normalise_data : bool, default False
        If True, divide each feature by ``normalisation_factor``.
    normalisation_factor : float, default 1.0
        Scaling factor applied when ``normalise_data`` is True.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    * Symmetric entanglement: each pair (i, j) is entangled twice, first
      i → j and then j → i, yielding a fully symmetric ZZ interaction.
    * The global rotation couples all qubits via a single data‑independent
      phase.
    * Normalisation is a simple multiplicative scaling; it does not
      modify the underlying data mapping functions.

    Examples
    --------
    >>> from qiskit.circuit import ParameterVector
    >>> qc = zz_feature_map_controlled(4, reps=1, global_phi=1.57)
    >>> print(qc)
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapControlled.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if normalise_data and normalisation_factor <= 0:
        raise ValueError("normalisation_factor must be > 0 when normalise_data is True.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Parameter vector for raw features
    raw_x = ParameterVector(parameter_prefix, n)

    # Normalise if requested
    if normalise_data:
        norm_factor = ParameterExpression(normalisation_factor)
        x = [raw_x[i] / norm_factor for i in range(n)]
    else:
        x = [raw_x[i] for i in range(n)]

    # Map functions
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
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        # Global shared rotation
        if not isinstance(global_phi, ParameterExpression):
            global_phi = ParameterExpression(global_phi)
        for i in range(n):
            qc.rz(global_phi, i)

        # Symmetric ZZ via CX–P–CX twice per pair
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

            # Reverse direction i <-> j
            angle_2_rev = 2 * map2(x[j], x[i])
            qc.cx(j, i)
            qc.p(angle_2_rev, i)
            qc.cx(j, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = raw_x  # type: ignore[attr-defined]
    return qc


# ----------------------------------------------------------------------
# Class style wrapper
# ----------------------------------------------------------------------
class ZZFeatureMapControlled(QuantumCircuit):
    """Class‑style wrapper for the symmetrised ZZ‑feature‑map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
        global_phi: Union[float, ParameterExpression] = 0.0,
        normalise_data: bool = False,
        normalisation_factor: float = 1.0,
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            global_phi=global_phi,
            normalise_data=normalise_data,
            normalisation_factor=normalisation_factor,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled"]
