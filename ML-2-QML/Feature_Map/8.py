"""Upgraded ZZFeatureMap with controlled‑modification scaling.

This module retains the canonical ZZFeatureMap interface while adding
the following optional extensions:

- ``use_rzz``: Replace the CX–P–CX ZZ implementation with a native
  ``RZZ`` gate.  The resulting unitary is mathematically identical
  when the same phase angle is supplied, but the RZZ gate is often
  more efficient on hardware that supports it natively.

- ``shared_pair_params``: When set, all pair‑wise interactions share
  a single parameter per entanglement pair.  This reduces the total
  number of trainable parameters and enforces symmetry across the
  circuit.

- ``normalise``: If ``True`` the helper mapping functions multiply
  the input features by ``π``.  This is a lightweight way to map
  arbitrary data into the natural domain of the phase gates
  without requiring external preprocessing.

The module exposes both a functional builder ``zz_feature_map`` and
a class‑based wrapper ``ZZFeatureMap`` for convenient OO usage.
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
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
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


def _default_map_1(x: ParameterExpression, normalise: bool = False) -> ParameterExpression:
    """Default φ₁(x) = x (optionally scaled to [0, π])."""
    return (pi if normalise else 1) * x


def _default_map_2(
    x: ParameterExpression,
    y: ParameterExpression,
    normalise: bool = False,
) -> ParameterExpression:
    """Default φ₂(x, y) = (π − x)(π − y) (optionally scaled)."""
    scale = pi if normalise else 1
    return (scale - x) * (scale - y)


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap with controlled‑modification extensions
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    use_rzz: bool = False,
    shared_pair_params: bool = False,
    normalise: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an enhanced ZZ‑feature‑map quantum circuit.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / data features. Must be ≥ 2.
    reps : int
        Number of repetitions of the feature‑map block. Must be > 0.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of which qubit pairs receive entangling gates.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        User‑supplied function that maps a slice of the data vector to a
        phase angle.  If ``None`` the default mappings are used.
    parameter_prefix : str
        Prefix used for the single‑qubit parameter vector.
    insert_barriers : bool
        Insert barriers between logical sub‑blocks for readability.
    use_rzz : bool
        If ``True`` replace the CX–P–CX ZZ implementation with a native
        ``RZZ`` gate.  The circuit remains mathematically equivalent.
    shared_pair_params : bool
        When ``True`` all pair‑wise entanglers share a single parameter
        per pair.  This reduces the parameter count and enforces symmetry.
    normalise : bool
        If ``True`` the default data‑mapping functions scale their
        arguments to the [0, π] interval.  This is a lightweight
        preprocessing step; full data normalisation should still be
        performed externally if required.
    name : str | None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    - The circuit is compatible with Qiskit’s data‑encoding workflows.
    - Parameter vectors are attached to the circuit via ``input_params``
      (single‑qubit) and ``pair_params`` (if ``shared_pair_params``).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    # Single‑qubit parameters
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Shared pair parameters, if requested
    pair_params: ParameterVector | None = None
    if shared_pair_params:
        pair_params = ParameterVector("p_pair", len(pairs))

    # Helper mapping functions
    if data_map_func is None:
        map1 = lambda xi: _default_map_1(xi, normalise=normalise)
        map2 = (
            lambda xi, xj, idx: pair_params[idx]
            if shared_pair_params
            else lambda xi, xj: _default_map_2(xi, xj, normalise=normalise)
        )
    else:
        map1 = lambda xi: data_map_func([xi])
        map2 = (
            lambda xi, xj, idx: pair_params[idx]
            if shared_pair_params
            else lambda xi, xj: data_map_func([xi, xj])
        )

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pair‑wise entanglement
        for idx, (i, j) in enumerate(pairs):
            angle_2 = 2 * (map2(x[i], x[j], idx) if shared_pair_params else map2(x[i], x[j]))
            if use_rzz:
                qc.rzz(angle_2, i, j)
            else:
                qc.cx(i, j)
                qc.p(angle_2, j)
                qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    if pair_params:
        qc.pair_params = pair_params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the enhanced ZZFeatureMap.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / data features. Must be ≥ 2.
    reps : int
        Number of repetitions of the feature‑map block. Must be > 0.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of which qubit pairs receive entangling gates.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        User‑supplied mapping function for data to phase angles.
    parameter_prefix : str
        Prefix used for the single‑qubit parameter vector.
    insert_barriers : bool
        Insert barriers between logical sub‑blocks for readability.
    use_rzz : bool
        Replace CX–P–CX with a native RZZ gate.
    shared_pair_params : bool
        Share a single parameter per entanglement pair.
    normalise : bool
        Scale default mapping functions to [0, π].
    name : str
        Name of the circuit.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        use_rzz: bool = False,
        shared_pair_params: bool = False,
        normalise: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            use_rzz=use_rzz,
            shared_pair_params=shared_pair_params,
            normalise=normalise,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        if hasattr(built, "pair_params"):
            self.pair_params = built.pair_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
