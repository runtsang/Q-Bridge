"""
SymmetricZZFeatureMap: A controlled‑modification of the canonical ZZ feature map.

This module provides a functional builder `symmetric_zz_feature_map` and a
`QuantumCircuit` subclass `SymmetricZZFeatureMap`.  The design introduces:
  • Shared single‑qubit and pairwise phase parameters that can be reused across
    layers (controlled modification).
  • Optional data normalisation to the range [0, π] via a user‑supplied
    `data_map_func` (default linear scaling).
  • Flexible entanglement patterns (full, linear, circular, or custom).
  • Barrier insertion for clearer visualisation.
  • Robust error handling and informative messages.

The circuit remains compatible with Qiskit’s data‑encoding workflow:
features are bound through the `input_params` attribute.
"""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Symmetric ZZ Feature Map
# ---------------------------------------------------------------------------

def symmetric_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    use_shared_params: bool = True,
    shared_layers: bool = True,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a symmetry‑aware ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input data. Must be >= 2.
    reps : int, optional
        Number of repetitions of the basic block. Default is 2.
    entanglement : str | sequence | callable, optional
        Defines which qubit pairs receive ZZ entanglement. See
        `_resolve_entanglement` for supported forms.
    data_map_func : callable, optional
        Function mapping a list of `ParameterExpression` to a single
        `ParameterExpression`. If None, defaults to the original
        φ1/φ2 mapping.
    parameter_prefix : str, optional
        Prefix for parameter names. Default is "x".
    use_shared_params : bool, optional
        If True, use a single parameter per qubit and a single parameter per
        pair across all repetitions (controlled modification). Default
        is True.
    shared_layers : bool, optional
        When `use_shared_params` is True, this flag controls whether the
        shared parameters are reused across layers (`True`) or each layer
        gets its own copy (`False`). Default is True.
    insert_barriers : bool, optional
        Insert barriers between logical blocks for visual clarity.
    name : str | None, optional
        Name of the resulting circuit. Defaults to "SymmetricZZFeatureMap".

    Returns
    -------
    QuantumCircuit
        A parameterised ZZ feature map circuit.

    Raises
    ------
    ValueError
        If `feature_dimension < 2` or if entanglement pairs are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for SymmetricZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "SymmetricZZFeatureMap")

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)
    pair_count = len(pairs)

    # Parameter vectors
    if use_shared_params:
        # Shared across layers if requested
        qubit_param_len = n if shared_layers else n * reps
        pair_param_len = pair_count if shared_layers else pair_count * reps
        qubit_params = ParameterVector(f"{parameter_prefix}_qubit", qubit_param_len)
        pair_params = ParameterVector(f"{parameter_prefix}_pair", pair_param_len)
    else:
        # Unique per qubit/pair per repetition
        qubit_params = ParameterVector(f"{parameter_prefix}_qubit", n * reps)
        pair_params = ParameterVector(f"{parameter_prefix}_pair", pair_count * reps)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Build repetitions
    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            idx = i if shared_layers else rep * n + i
            qc.p(2 * map1(qubit_params[idx]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglement via CX–P–CX
        for (i, j) in pairs:
            idx = pairs.index((i, j)) if shared_layers else rep * pair_count + pairs.index((i, j))
            angle_2 = 2 * map2(qubit_params[i], qubit_params[j]) if not use_shared_params else 2 * map2(pair_params[idx], pair_params[idx])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach input parameters for easy binding
    qc.input_params = qubit_params if use_shared_params else qubit_params
    return qc


class SymmetricZZFeatureMap(QuantumCircuit):
    """Object‑oriented wrapper for the SymmetricZZFeatureMap.

    The constructor builds the circuit via :func:`symmetric_zz_feature_map`
    and composes it into the instance.  The resulting object exposes the
    same `input_params` attribute for parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        use_shared_params: bool = True,
        shared_layers: bool = True,
        insert_barriers: bool = False,
        name: str = "SymmetricZZFeatureMap",
    ) -> None:
        built = symmetric_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            use_shared_params=use_shared_params,
            shared_layers=shared_layers,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["SymmetricZZFeatureMap", "symmetric_zz_feature_map"]
