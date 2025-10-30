"""Qiskit-compatible controlled‑modification ZZ feature map.

This module implements a controlled‑modification of the canonical ZZFeatureMap.
The circuit follows the structure:

    H → P(2·φ1(x)) on each qubit
    → symmetric ZZ entanglers via CX–P–CX applied in both directions
    repeated for ``reps`` times.

Controlled modifications:

* Symmetric pair entanglement: each entangling pair (i, j) is applied twice,
  once with i as control and once with j as control, ensuring a fully
  symmetric interaction graph.
* Global entanglement strength parameter γ: a single parameter multiplies
  all pair angles, allowing the ZZ coupling to be tuned globally.
* Optional feature scaling via ``feature_scale``.
* Optional barriers between logical blocks.

The module exposes a functional builder ``zz_feature_map_controlled`` and
an OO wrapper ``ZZFeatureMapControlled`` that can be instantiated directly.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

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

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        Entanglement specification.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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

# ---------------------------------------------------------------------------
# Feature map builder
# ---------------------------------------------------------------------------

def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    gamma_prefix: str = "gamma",
    feature_scale: float = 1.0,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a controlled‑modification ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension
        Number of classical features / qubits.
    reps
        Number of repetitions of the encoding block.
    entanglement
        Entanglement specification (see :func:`_resolve_entanglement`).
    data_map_func
        Optional callable that maps a list of parameters to an angle.
        If ``None`` the default mappings are used.
    parameter_prefix
        Prefix for the single‑qubit feature parameters.
    gamma_prefix
        Prefix for the global entanglement parameter.
    feature_scale
        Multiplicative scaling applied to every classical feature before
        it is mapped to a rotation angle.  Default is ``1.0`` (no scaling).
    insert_barriers
        If ``True`` insert a barrier after each logical block.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1 for ZZFeatureMapControlled.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if feature_scale <= 0:
        raise ValueError("feature_scale must be positive.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Parameter vectors
    x = ParameterVector(parameter_prefix, n)
    gamma = ParameterVector(gamma_prefix, 1)

    # Default angle mappings
    def default_map1(xi: ParameterExpression) -> ParameterExpression:
        """Default φ1(x) = x."""
        return xi * feature_scale

    def default_map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Default φ2(x, y) = (π − x)(π − y)."""
        return (pi - xi) * (pi - xj)

    # Resolve data mapping
    if data_map_func is None:
        map1 = default_map1
        map2 = default_map2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi]) * feature_scale

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj]) * feature_scale

    # Entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Symmetric ZZ entanglers
        for (i, j) in pairs:
            # Global strength γ multiplies the pair angle
            angle = 2 * map2(x[i], x[j]) * gamma[0]
            # i → j
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
            # j → i
            qc.cx(j, i)
            qc.p(angle, i)
            qc.cx(j, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Expose parameters for binding
    qc.input_params = [x, gamma]  # type: ignore[attr-defined]
    return qc

# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapControlled(QuantumCircuit):
    """Object‑oriented wrapper for :func:`zz_feature_map_controlled`."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        gamma_prefix: str = "gamma",
        feature_scale: float = 1.0,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            gamma_prefix=gamma_prefix,
            feature_scale=feature_scale,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        # Preserve input parameters for binding
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled"]
