"""Extended ZZFeatureMap builder with higher‑order interactions, optional pre/post rotations,
adaptive depth, and data scaling.

This module extends the canonical ZZFeatureMap by adding:
- multi‑qubit ZZ interactions up to a user‑defined order.
- optional Rx before the Hadamard layer and Ry after the entanglement layer.
- adaptive repetition count based on the feature dimension.
- data‑scaling factor and optional data mapping function.
- barrier insertion for easier circuit inspection.

Usage:
>>> from zz_feature_map_extension import zz_feature_map, ZZFeatureMap
>>> qc = zz_feature_map(feature_dimension=4, interaction_order=3, pre_rotations=True)
>>> qc.draw()
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
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


def _resolve_interactions(
    num_qubits: int,
    order: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int,...]]:
    """Return a list of tuples describing multi‑qubit interactions.

    For ``order == 2`` the result is the same as ``_resolve_entanglement``.
    For higher orders the function generates all combinations that respect the
    base entanglement pattern. The default behaviour is to use a sliding window
    of size ``order`` over the qubit indices, which is sufficient for most
    classification tasks and keeps the circuit depth modest.
    """
    if order < 2:
        raise ValueError("interaction_order must be >= 2")

    if order == 2:
        return [tuple(pair) for pair in _resolve_entanglement(num_qubits, entanglement)]

    # Sliding window of size `order`, each window is a tuple of indices
    windows: List[Tuple[int,...]] = []
    for start in range(num_qubits - order + 1):
        windows.append(tuple(range(start, start + order)))

    # If user supplied a custom entanglement list, we try to extend it
    # by appending the last qubit of each pair as the new qubit.
    if isinstance(entanglement, Sequence) and all(len(p) == 2 for p in entanglement):
        for (i, j) in entanglement:
            if j + 1 < num_qubits:
                windows.append((i, j, j + 1))

    return windows


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Extended ZZFeatureMap
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int | None = None,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    interaction_order: int = 2,
    pre_rotations: bool = False,
    post_rotations: bool = False,
    data_scaling: float = 1.0,
    adaptive_depth: bool = False,
    use_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑feature‑map circuit.

    The circuit follows the canonical structure but adds several new knobs:

    * **interaction_order** – include ZZ, ZZZ, … up to the user‑supplied order.
    * **pre_rotations / post_rotations** – optional Rx/ Ry layers before and after the entanglement.
    * **adaptive_depth** – if ``True``, the number of repetitions is set to ``max(1, n//2)``.
    * **data_scaling** – global multiplier applied to all rotation angles.
    * **use_barriers** – insert barriers for visual clarity.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int | None, optional
        Number of repetitions. Ignored if ``adaptive_depth`` is ``True``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specifies the two‑qubit entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        User‑supplied function mapping feature values to rotation angles.
    parameter_prefix : str, optional
        Prefix for the parameter vector.
    interaction_order : int, optional
        Highest‑order interaction to include (>=2).
    pre_rotations : bool, optional
        If ``True``, an Rx(π/2) gate is applied before the Hadamard layer.
    post_rotations : bool, optional
        If ``True``, an Ry(π/2) gate is applied after the entanglement layer.
    data_scaling : float, optional
        Multiplier applied to all data‑derived angles.
    adaptive_depth : bool, optional
        If ``True``, ``reps`` is ignored and set to ``max(1, n//2)``.
    use_barriers : bool, optional
        Insert barriers for easier debugging.
    name : str | None, optional
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit ready for binding.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``interaction_order`` < 2, or if the
        interaction order is larger than the number of qubits.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if interaction_order < 2:
        raise ValueError("interaction_order must be >= 2.")
    if interaction_order > feature_dimension:
        raise ValueError(
            f"interaction_order ({interaction_order}) cannot exceed the number of qubits "
            f"({feature_dimension})."
        )

    n = int(feature_dimension)
    if reps is None:
        reps = 2
    if adaptive_depth:
        reps = max(1, n // 2)

    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    x = ParameterVector(parameter_prefix, n)

    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:

        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_interactions(n, interaction_order, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotations
        if pre_rotations:
            qc.rx(pi / 2, range(n))
            if use_barriers:
                qc.barrier()

        # Basis prep
        qc.h(range(n))
        if use_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * data_scaling * map1(x[i]), i)
        if use_barriers:
            qc.barrier()

        # Entanglement via multi‑qubit ZZ operations
        for interaction in pairs:
            # Build a chain of CX gates that maps the first qubit to the last
            # Apply P on the last qubit, then reverse the chain.
            chain = list(interaction)
            if len(chain) < 2:
                continue  # nothing to entangle
            # forward CX chain
            for idx in range(len(chain) - 1):
                qc.cx(chain[idx], chain[idx + 1])
            # Phase on the last qubit
            angle_2 = 2 * data_scaling * map2(x[chain[0]], x[chain[-1]])
            qc.p(angle_2, chain[-1])
            # reverse CX chain
            for idx in reversed(range(len(chain) - 1)):
                qc.cx(chain[idx], chain[idx + 1])

        if use_barriers:
            qc.barrier()

        # Optional post‑rotations
        if post_rotations:
            qc.ry(pi / 2, range(n))
            if use_barriers:
                qc.barrier()

        if use_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the extended ZZFeatureMap.

    Parameters
    ----------
    All parameters are forwarded to :func:`zz_feature_map`.  See its
    documentation for details.

    Attributes
    ----------
    input_params : ParameterVector
        Parameters that need to be bound to a classical data vector.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int | None = None,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        interaction_order: int = 2,
        pre_rotations: bool = False,
        post_rotations: bool = False,
        data_scaling: float = 1.0,
        adaptive_depth: bool = False,
        use_barriers: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            interaction_order=interaction_order,
            pre_rotations=pre_rotations,
            post_rotations=post_rotations,
            data_scaling=data_scaling,
            adaptive_depth=adaptive_depth,
            use_barriers=use_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
