"""
Shared module for a controlled‑modified ZZ‑type feature map.

Features
--------
* Symmetrised all‑to‑all ZZ couplings with a uniform phase shift.
* Normalised quadratic data mapping: φ1(x)=x/π, φ2(x,y)=x*y/π.
* Global scaling factor applied to all angles.
* Optional entanglement patterns: full, linear, circular, explicit list.
* Parameter binding compatible with Qiskit’s data‑encoding workflows.
* Both a functional helper (zz_feature_map_scaled) and a class wrapper (ZZFeatureMapScaled).

Usage
-----
>>> from zz_feature_map_scaled import zz_feature_map_scaled, ZZFeatureMapScaled
>>> qc = zz_feature_map_scaled(feature_dimension=3, reps=1)
>>> qc.draw()
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit mapping: φ1(x) = x / π."""
    return x / pi


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit mapping: φ2(x, y) = (x * y) / π."""
    return (x * y) / pi


def zz_feature_map_scaled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    scaling_factor: float | ParameterExpression = 1.0,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a controlled‑modified ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, default 2
        Number of repetitions of the base pattern.
    entanglement : str or sequence of pairs or callable, default "full"
        Entanglement specification (see :func:`_resolve_entanglement`).
    data_map_func : callable, optional
        Function mapping a list of :class:`~qiskit.circuit.ParameterExpression`
        to a single :class:`~qiskit.circuit.ParameterExpression`. If None,
        the default quadratic maps are used.
    parameter_prefix : str, default "x"
        Prefix for the :class:`~qiskit.circuit.ParameterVector` parameters.
    scaling_factor : float or ParameterExpression, default 1.0
        Global factor applied to all data‑driven angles.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for visual clarity.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    The circuit is compatible with Qiskit’s data‑encoding workflows:
    the returned circuit exposes a ``.input_params`` attribute containing
    the :class:`~qiskit.circuit.ParameterVector` that can be bound to a
    numpy array of shape (feature_dimension,).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapScaled.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapScaled")
    x = ParameterVector(parameter_prefix, n)

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
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * scaling_factor * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            angle_2 = 2 * scaling_factor * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapScaled(QuantumCircuit):
    """
    Class‑style wrapper for the controlled‑modified ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, default 2
        Number of repetitions of the base pattern.
    entanglement : str or sequence of pairs or callable, default "full"
        Entanglement specification.
    data_map_func : callable, optional
        Custom data‑mapping function.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector parameters.
    scaling_factor : float or ParameterExpression, default 1.0
        Global factor applied to all data‑driven angles.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    name : str, default "ZZFeatureMapScaled"
        Circuit name.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        scaling_factor: float | ParameterExpression = 1.0,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapScaled",
    ) -> None:
        built = zz_feature_map_scaled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            scaling_factor=scaling_factor,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapScaled", "zz_feature_map_scaled"]
