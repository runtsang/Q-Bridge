"""ZZFeatureMapRZZControlled: Controlled modification of the RZZ entangler feature map."""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to the entanglement spec.

    Supported specs:
        - "full": all-to-all pairs (i < j)
        - "linear": nearest neighbours
        - "circular": linear plus wrap‑around
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs

# --------------------------------------------------------------------------- #
# Default data mapping functions
# --------------------------------------------------------------------------- #
def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x

def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)

# --------------------------------------------------------------------------- #
# Optional feature‑vector normalisation helper
# --------------------------------------------------------------------------- #
def normalize_features(features: Sequence[float]) -> List[float]:
    """Return a L2‑normalised copy of *features*.

    Parameters
    ----------
    features : Sequence[float]
        Raw feature vector.

    Returns
    -------
    List[float]
        Normalised vector with unit Euclidean norm.
    """
    vec = np.asarray(features, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalise a zero vector.")
    return vec / norm

# --------------------------------------------------------------------------- #
# Feature map builder
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    shared_interaction_scale: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Symmetrised RZZ feature map with a shared interaction parameter.

    Parameters
    ----------
    feature_dimension : int
        Number of input features (must be >= 2).
    reps : int, optional
        Number of repetitions of the primitive pattern. Default is 2.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement specification. See :func:`_resolve_entanglement` for details.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Custom mapping from raw features to rotation angles. If ``None`` the
        defaults φ1 and φ2 (see below) are used.
    parameter_prefix : str, optional
        Prefix for the symbolic parameters. Default is ``"x"``.
    insert_barriers : bool, optional
        Whether to insert barriers between layers for visual clarity.
    shared_interaction_scale : float, optional
        Global scaling factor applied to every RZZ interaction.  This
        parameter is *independent* of the data and can be tuned
        during training.
    name : str | None, optional
        Circuit name.  If ``None`` a default is supplied.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with a feature vector.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    # Build the parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Determine mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        # data_map_func must accept a list of ParameterExpressions
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Construct the circuit
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        # Symmetric pairwise RZZ with shared scale
        for (i, j) in pairs:
            angle = 2 * shared_interaction_scale * map2(x[i], x[j])
            qc.rzz(angle, i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# OO wrapper
# --------------------------------------------------------------------------- #
class ZZFeatureMapRZZControlled(QuantumCircuit):
    """
    Object‑oriented wrapper for :func:`zz_feature_map_rzz_controlled`.

    Parameters are identical to the functional builder.  The wrapper
    stores the input parameter vector as ``self.input_params`` for
    easy binding and compatibility with Qiskit's data‑encoding tools.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        shared_interaction_scale: float = 1.0,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            shared_interaction_scale,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
