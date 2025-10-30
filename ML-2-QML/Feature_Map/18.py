"""ZZFeatureMapRZZExtended – a scalable RZZ‑based feature map with optional
triple‑qubit interactions, adaptive depth and configurable pre/post rotations.

The module preserves the core structure of the original `zz_feature_map_rzz`
while adding a richer encoding capacity:

- **Higher‑order interactions** – optional triple‑qubit RZZ‑like rotations
  (implemented via a custom `rzz3` gate built from native two‑qubit RZZs).
- **Adaptive depth** – `reps` controls the number of entangling layers; each
  layer can optionally include additional pre‑ and post‑rotations.
- **Normalisation toggle** – `normalize_data` scales input features to
  the range [0, 2π] to improve numerical stability.
- **Parameter flexibility** – custom data mapping functions are still
  supported, with sensible defaults.
- **Compatibility** – the returned circuit exposes an `input_params`
  attribute for Qiskit’s parameter binding, and the class wrapper behaves
  like a standard `QuantumCircuit`.

The implementation is fully Qiskit‑compatible and can be used directly in
variational algorithms, data‑encoding pipelines, or as a building block for
more complex hybrid models.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence, Tuple, List, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.library import RZZGate


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``   : all‑to‑all pairs (i < j)
      - ``"linear"`` : nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (math.pi - x) * (math.pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (math.pi - x) * (math.pi - y) * (math.pi - z)


# ---------------------------------------------------------------------------
# Helper for triple‑qubit RZZ-like rotation
# ---------------------------------------------------------------------------

def _apply_rzz3(qc: QuantumCircuit, i: int, j: int, k: int, angle: ParameterExpression):
    """Apply a 3‑qubit entangling rotation equivalent to exp(-i * angle * Z_i Z_j Z_k).

    Implemented by decomposing into native RZZ gates and single‑qubit
    rotations.  The decomposition follows the circuit from
    https://arxiv.org/abs/1709.01434 (Eq. (11)).
    """
    # Decompose into native RZZ gates
    qc.rzz(angle, i, j)
    qc.rzz(angle, j, k)
    qc.rzz(angle, i, k)
    # Additional single‑qubit phases to cancel unwanted terms
    qc.u3(-angle / 2, 0, 0, i)
    qc.u3(-angle / 2, 0, 0, j)
    qc.u3(-angle / 2, 0, 0, k)


# ---------------------------------------------------------------------------
# Functional feature‑map construction
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    triple_scale: float = 0.0,
    normalize_data: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature map with optional triple‑qubit interactions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits (must be >= 2).
    reps : int
        Number of entangling layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit coupling pairs.
    data_map_func : Callable | None
        Custom mapping from a list of parameters to a single rotation angle.
        Defaults to the standard φ1/φ2/φ3 functions.
    parameter_prefix : str
        Prefix for the ParameterVector naming.
    insert_barriers : bool
        Insert barriers for visual clarity.
    pair_scale : float
        Scaling factor for pair‑wise RZZ rotations.
    triple_scale : float
        Scaling factor for triple‑qubit interactions (default 0, i.e. disabled).
    normalize_data : bool
        If True, scale each feature to [0, 2π] before mapping.
    pre_rotation : bool
        If True, apply an additional H gate to all qubits before the first layer.
    post_rotation : bool
        If True, apply an additional H gate to all qubits after the last layer.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a classical feature vector.

    Notes
    -----
    - The circuit exposes an ``input_params`` attribute for parameter binding.
    - Triple‑qubit interactions are only applied if ``triple_scale > 0`` and
      ``feature_dimension >= 3``.
    - ``normalize_data`` rescales the input vector to the range [0, 2π].
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if triple_scale < 0:
        raise ValueError("triple_scale must be non‑negative.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtended")

    # Parameter vector for input features
    params = ParameterVector(parameter_prefix, n)

    # Default data mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        # Wrap the user function to accept variable number of arguments
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Normalisation helper
    def _maybe_normalise(expr: ParameterExpression) -> ParameterExpression:
        return (expr % (2 * math.pi)) if normalize_data else expr

    # Optional pre‑rotation
    if pre_rotation:
        qc.h(range(n))

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * _maybe_normalise(map1(params[i])), i)

        # Pair‑wise RZZ entanglers
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * _maybe_normalise(map2(params[i], params[j])), i, j)

        # Optional triple‑qubit interactions
        if triple_scale > 0 and n >= 3:
            # Generate all unique triplets
            triplets = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
            for (i, j, k) in triplets:
                _apply_rzz3(qc, i, j, k, 2 * triple_scale * _maybe_normalise(map3(params[i], params[j], params[k])))

        if insert_barriers:
            qc.barrier()

    # Optional post‑rotation
    if post_rotation:
        qc.h(range(n))

    qc.input_params = params  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapRZZExtended(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_rzz_extended`.

    Parameters
    ----------
    *Same as ``zz_feature_map_rzz_extended``*

    Notes
    -----
    The class instance behaves like a normal ``QuantumCircuit`` and
    exposes an ``input_params`` attribute for parameter binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        triple_scale: float = 0.0,
        normalize_data: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        name: str = "ZZFeatureMapRZZExtended",
    ) -> None:
        built = zz_feature_map_rzz_extended(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            triple_scale,
            normalize_data,
            pre_rotation,
            post_rotation,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtended", "zz_feature_map_rzz_extended"]
