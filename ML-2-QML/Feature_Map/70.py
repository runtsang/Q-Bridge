"""ZZFeatureMapRZZControlled – a controlled‑modification variant of the RZZ‑based feature map.

The module defines two interfaces:
- `zz_feature_map_rzz_controlled` – functional API that returns a QuantumCircuit.
- `ZZFeatureMapRZZControlled` – a class inheriting from QuantumCircuit.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    n: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return the entanglement pairs for the circuit.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors
      - "circular": linear plus wrap‑around
      - explicit list of pairs
      - callable: f(n) -> sequence of (i, j)
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(n))
        return [(int(i), int(j)) for (i, j) in pairs]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={n}.")
    return pairs

# --------------------------------------------------------------------------- #
# Default mapping functions (support interaction_order)
# --------------------------------------------------------------------------- #

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x

def _default_map_2(
    x_i: ParameterExpression,
    x_j: ParameterExpression,
    interaction_order: str = "quadratic",
) -> ParameterExpression:
    """Default φ2(xi, xj) depends on interaction_order.

    - "linear": (xi + xj) / 2
    - "quadratic": xi * xj
    - "cubic": xi * xj * (xi + xj) / 2
    """
    if interaction_order == "linear":
        return (x_i + x_j) / 2
    if interaction_order == "quadratic":
        return x_i * x_j
    if interaction_order == "cubic":
        return x_i * x_j * (x_i + x_j) / 2
    raise ValueError(f"Unsupported interaction_order: {interaction_order!r}")

# --------------------------------------------------------------------------- #
# Functional API
# --------------------------------------------------------------------------- #

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    interaction_order: str = "quadratic",
    normalise: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Return a Qiskit QuantumCircuit implementing a controlled‑modification ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        How to entangle qubits.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional custom mapping from raw parameters to angles.
    parameter_prefix : str
        Prefix for the symbolic parameters.
    insert_barriers : bool
        Insert barriers after each block for readability.
    pair_scale : float
        Scaling factor for the pair interaction angles.
    interaction_order : str
        One of "linear", "quadratic", or "cubic" to choose the default interaction function.
    normalise : bool
        If True, the user is expected to normalise the data before binding.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data binding.

    Notes
    -----
    The circuit structure:
      1. H on all qubits
      2. P(2 * φ1(x_i)) on each qubit
      3. RZZ(2 * pair_scale * φ2(x_i, x_j)) on each entangled pair
    The default φ1 and φ2 are defined by ``_default_map_1`` and ``_default_map_2``.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    if interaction_order not in {"linear", "quadratic", "cubic"}:
        raise ValueError(f"interaction_order must be one of 'linear', 'quadratic', 'cubic'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return _default_map_1(xi)

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return _default_map_2(xi, xj, interaction_order)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# Class-style API
# --------------------------------------------------------------------------- #

class ZZFeatureMapRZZControlled(QuantumCircuit):
    """QuantumCircuit subclass for the controlled‑modification ZZ feature map.

    Parameters are identical to :func:`zz_feature_map_rzz_controlled`.
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
        interaction_order: str = "quadratic",
        normalise: bool = False,
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
            interaction_order,
            normalise,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
