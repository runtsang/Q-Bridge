"""Controlled‑modification ZZFeatureMap with symmetrised interactions and global phase offset.

This module provides a functional helper `zz_feature_map` and an OO wrapper `ZZFeatureMap`.
Both expose the same public API as the original canonical implementation but add:

* **symmetrized interaction mapping** – ensures φ₂(xᵢ, xⱼ) = φ₂(xⱼ, xᵢ) by averaging user‑supplied maps.
* **global phase offset** – adds a constant rotation to all single‑qubit phases.
* **customizable entanglement** – supports ‘full’, ‘linear’, ‘circular’, explicit lists, or callables.
* **optional barriers** – for visualising circuit stages.

The module remains fully compatible with Qiskit's data‑encoding workflows: the returned `QuantumCircuit` has an `input_params` attribute that can be bound to a NumPy array of shape `(feature_dimension,)`.

Authoritative, succinct, and technically precise design choices are documented in the function and class docstrings.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

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
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
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
    """Default φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap (CX–P–CX for ZZ) – controlled‑modification version
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    *,
    symmetrize: bool = False,
    global_phase_offset: float = 0.0,
) -> QuantumCircuit:
    """Build a controlled‑modification ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 1.
    reps : int, default 2
        Number of repetitions of the feature‑map layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : callable or None, default None
        User‑supplied mapping from a list of parameters to a single rotation angle.
        If None, defaults to the canonical φ₁ and φ₂ definitions.
    parameter_prefix : str, default "x"
        Prefix for the `ParameterVector` naming.
    insert_barriers : bool, default False
        Insert barriers after each major block for visual clarity.
    name : str | None, default None
        Optional circuit name.
    symmetrize : bool, default False
        If True, enforce φ₂(xᵢ, xⱼ) = φ₂(xⱼ, xᵢ) by averaging the two evaluations.
    global_phase_offset : float, default 0.0
        Constant rotation added to every single‑qubit phase (in radians).

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data binding.

    Notes
    -----
    * The circuit uses the CX–P–CX construction for ZZ interactions.
    * The function validates input bounds and raises informative errors.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1 for ZZFeatureMap.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    # Parameter vector for the raw data
    x = ParameterVector(parameter_prefix, n)

    # Resolve the data mapping functions
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

        # Single‑qubit phases with optional global offset
        for i in range(n):
            angle = 2 * map1(x[i]) + 2 * global_phase_offset
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglement via CX–P–CX
        for (i, j) in pairs:
            if symmetrize:
                # Average the two possible orderings
                angle_2 = 2 * ((map2(x[i], x[j]) + map2(x[j], x[i])) / 2)
            else:
                angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification ZZFeatureMap.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 1.
    reps : int, default 2
        Number of repetitions of the feature‑map layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : callable or None, default None
        User‑supplied mapping from a list of parameters to a single rotation angle.
    parameter_prefix : str, default "x"
        Prefix for the `ParameterVector` naming.
    insert_barriers : bool, default False
        Insert barriers after each major block for visual clarity.
    name : str, default "ZZFeatureMap"
        Circuit name.
    symmetrize : bool, default False
        Enforce symmetric interaction mapping.
    global_phase_offset : float, default 0.0
        Constant rotation added to every single‑qubit phase (in radians).
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMap",
        *,
        symmetrize: bool = False,
        global_phase_offset: float = 0.0,
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            symmetrize=symmetrize,
            global_phase_offset=global_phase_offset,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
