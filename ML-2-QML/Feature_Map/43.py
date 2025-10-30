"""Symmetrised ZZ feature map with optional shared phase and normalisation.

This module extends the canonical ZZFeatureMap by adding:
- Symmetric entanglement: each ZZ coupling is applied twice (CX–P–CX) with the
  phase on the target qubit set to the average of the two feature values.
- Optional shared global phase that depends on the sum of all feature angles.
- Optional normalisation of feature angles by π.
- Flexible data‑mapping function that can be overridden for custom feature
  engineering while still using the same structural skeleton.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple, List, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a user‑supplied entanglement specification into an explicit list of
    two‑qubit pairs.

    Supported string specifiers
    ---------------------------
    - ``"full"``   : all‑to‑all pairs (i < j)
    - ``"linear"`` : nearest‑neighbour chain
    - ``"circular"``: linear plus wrap‑around for n > 2

    A callable or an explicit sequence can also be supplied.

    Raises
    ------
    ValueError
        If an unrecognised string is given or a pair contains identical qubits
        or out‑of‑range indices.
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
        return [(int(i), int(j)) for (i, j) in entanglement(num_qubits)]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Symmetrised ZZFeatureMap (CX–P–CX with averaged angles)
# ---------------------------------------------------------------------------

def zz_feature_map_sym(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    shared_phase: bool = False,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a symmetrised ZZ‑feature‑map circuit.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default=2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit coupling pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional custom mapping from raw feature values to rotation angles.
        If ``None`` a simple average mapping is used.
    parameter_prefix : str, default="x"
        Prefix for automatically generated parameter names.
    insert_barriers : bool, default=False
        Insert barriers between logical blocks for easier circuit inspection.
    shared_phase : bool, default=False
        If ``True`` a global phase equal to the sum of all feature angles is
        applied to every qubit after the single‑qubit rotations.
    normalize : bool, default=False
        If ``True`` each feature value is divided by ``π`` before being used
        as a rotation angle.  This keeps all angles in the range
        ``[-1, 1]``.
    name : str | None, default=None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding to a feature vector.

    Notes
    -----
    - The entanglement is *symmetrised*: for each pair (i, j) the phase on
      qubit ``j`` is set to the average ``(x_i + x_j)/2``.  This preserves
      the ZZ interaction while making the circuit more robust to noise.
    - A shared global phase can be toggled, which may be useful for
      embedding global information or for symmetry‑based feature
      engineering.
    - The optional ``data_map_func`` lets users inject arbitrary
      non‑linear transformations (e.g. sin, cos) while still using the
      same structural skeleton.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2 or ``reps`` < 1.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapSym.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapSym")

    x = ParameterVector(parameter_prefix, n)

    # Mapping functions
    if data_map_func is None:
        if normalize:
            def map1(xi: ParameterExpression) -> ParameterExpression:
                return xi / pi
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return (xi + xj) / (2 * pi)
        else:
            def map1(xi: ParameterExpression) -> ParameterExpression:
                return xi
            def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
                return (xi + xj) / 2
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

        # Optional shared global phase
        if shared_phase:
            global_angle = sum(map1(xi) for xi in x)  # type: ignore[assignment]
            for i in range(n):
                qc.p(global_angle, i)

        if insert_barriers:
            qc.barrier()

        # Symmetric ZZ entanglement via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapSym(QuantumCircuit):
    """Object‑oriented wrapper for the symmetrised ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default=2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit coupling pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default=None
        Optional custom mapping from raw feature values to rotation angles.
    parameter_prefix : str, default="x"
        Prefix for automatically generated parameter names.
    insert_barriers : bool, default=False
        Insert barriers between logical blocks for easier circuit inspection.
    shared_phase : bool, default=False
        If ``True`` a global phase equal to the sum of all feature angles is
        applied to every qubit after the single‑qubit rotations.
    normalize : bool, default=False
        If ``True`` each feature value is divided by ``π`` before being used
        as a rotation angle.
    name : str, default="ZZFeatureMapSym"
        Optional circuit name.
    """
    def __init__(self,
                 feature_dimension: int,
                 reps: int = 2,
                 entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
                 data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
                 parameter_prefix: str = "x",
                 insert_barriers: bool = False,
                 shared_phase: bool = False,
                 normalize: bool = False,
                 name: str = "ZZFeatureMapSym") -> None:
        built = zz_feature_map_sym(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            shared_phase=shared_phase,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapSym", "zz_feature_map_sym"]
