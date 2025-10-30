"""ZZFeatureMapRZZControlled: Controlled modification of the RZZ feature map.

Features:
- Shared global pair angle across all entangling gates.
- Symmetric data mapping for single‑qubit rotations and pairwise interactions.
- Optional normalisation of the input feature vector (user must normalise externally).
- Supports full, linear, circular, or custom entanglement patterns.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
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
# Default mapping functions
# --------------------------------------------------------------------------- #

def _default_phi1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit mapping φ₁(x) = x."""
    return x


def _default_phi2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default pairwise mapping φ₂(x, y) = (π − x)(π − y). Symmetric."""
    return (math.pi - x) * (math.pi - y)


def _default_pair_map(xs: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default global pair‑angle mapping: average of (π − xᵢ) over all features."""
    if not xs:
        raise ValueError("Feature vector must contain at least one element.")
    # Sum (π - xi) over all features and normalise by number of features
    avg_term = sum((math.pi - xi) for xi in xs) / len(xs)
    return avg_term


# --------------------------------------------------------------------------- #
# Functional implementation
# --------------------------------------------------------------------------- #

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    pair_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a ZZ‑style feature map using native RZZ gates with a globally shared pair angle.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (also number of qubits).
    reps : int, default 2
        Number of repetitions of the feature‑map layer.
    entanglement : str | sequence | callable, default "full"
        Entanglement pattern. See ``_resolve_entanglement`` for details.
    data_map_func : callable, optional
        User‑supplied mapping from a list of parameters to a single parameter
        expression.  If ``len(args) == 1`` it is used as φ₁; if ``len(args) == 2``
        it is used as φ₂; otherwise it is ignored.
    pair_map_func : callable, optional
        User‑supplied mapping from the full feature vector to a single
        parameter expression that will be used as the shared pair angle.
    parameter_prefix : str, default "x"
        Prefix for the single‑qubit parameters.
    insert_barriers : bool, default False
        Insert barriers between layers for visual clarity.
    pair_scale : float, default 1.0
        Global scaling factor applied to the pair‑angle.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A circuit ready for parameter binding.

    Notes
    -----
    The circuit supports parameter binding via ``circuit.bind_parameters``.  The
    single‑qubit parameters are stored in ``circuit.input_params`` and the
    global pair angle is stored in ``circuit.global_pair_angle``.
    """

    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(pair_scale, (int, float)):
        raise TypeError("pair_scale must be a numeric value.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Single‑qubit parameters
    x = ParameterVector(parameter_prefix, n)

    # Global pair‑angle parameter
    theta_pair = ParameterVector("theta_pair", 1)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Define mapping functions
    if data_map_func is None:
        phi1 = _default_phi1
        phi2 = _default_phi2
    else:
        def phi1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def phi2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    if pair_map_func is None:
        def pair_map(xs: Sequence[ParameterExpression]) -> ParameterExpression:
            return pair_scale * _default_pair_map(xs)
    else:
        def pair_map(xs: Sequence[ParameterExpression]) -> ParameterExpression:
            return pair_scale * pair_map_func(xs)

    # Compute the shared pair angle once per layer
    pair_angle_expr = pair_map(x)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * phi1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_angle_expr, i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach metadata
    qc.input_params = x  # type: ignore[attr-defined]
    qc.global_pair_angle = theta_pair[0]  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Object‑oriented wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapRZZControlled(QuantumCircuit):
    """
    OO wrapper around :func:`zz_feature_map_rzz_controlled`.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / classical features.
    reps : int, default 2
    entanglement : str | sequence | callable, default "full"
    data_map_func : callable, optional
    pair_map_func : callable, optional
    parameter_prefix : str, default "x"
    insert_barriers : bool, default False
    pair_scale : float, default 1.0
    name : str, default "ZZFeatureMapRZZControlled"

    Attributes
    ----------
    input_params : ParameterVector
        Single‑qubit parameters.
    global_pair_angle : ParameterExpression
        Global pair‑angle parameter.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        pair_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
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
            pair_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.global_pair_angle = built.global_pair_angle  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
