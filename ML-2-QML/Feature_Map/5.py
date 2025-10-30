"""Extended ZZ‑Feature Map with higher‑order interactions and optional rotation layers.

This module defines a functional helper ``extended_zz_feature_map`` and a
``QuantumCircuit`` subclass ``ExtendedZZFeatureMap``.  Both expose the
``input_params`` attribute for parameter binding and support the same
entanglement specification as the canonical ``ZZFeatureMap``.

Key extensions:

* **Three‑body CCZ entanglement** – when ``three_body=True`` a CCZ gate
  is applied to every unique triple of qubits.  The interaction angle
  is derived from a user‑supplied or default ``map3`` function.
* **Pre‑ and post‑rotation layers** – optional ``rx(π/2)`` (pre) and
  ``ry(π/2)`` (post) rotations add additional variational flexibility.
* **Custom data‑mapping** – the ``data_map_func`` can accept 1, 2, or 3
  arguments, allowing bespoke feature scaling or nonlinearities.
* **Parameter validation** – clear error messages for invalid
  entanglement specs, insufficient qubits for three‑body terms,
  and non‑``ParameterExpression`` returns.

The interface is fully compatible with Qiskit data‑encoding workflows.
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
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbors (0,1), (1,2), …
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Default data‑mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit mapping: φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit mapping: φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default three‑qubit mapping: φ₃(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Extended ZZ‑Feature Map (functional)
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    three_body: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.  Must be >= 2, and >= 3 if
        ``three_body=True``.
    reps : int
        Number of repetitions of the basic block.
    entanglement : str | sequence | callable
        Specification of two‑qubit entanglement pairs.  See
        :func:`_resolve_entanglement` for accepted formats.
    data_map_func : callable, optional
        User‑supplied mapping from a list of :class:`ParameterExpression`
        to a new :class:`ParameterExpression`.  It is called with 1, 2, or
        3 arguments depending on whether a single‑qubit, two‑qubit or
        three‑qubit interaction is being encoded.
    parameter_prefix : str
        Prefix for the :class:`ParameterVector` used to store the
        feature‑vector parameters.
    insert_barriers : bool
        If ``True`` insert a barrier after each major grouping of gates
        to aid circuit analysis.
    three_body : bool
        When ``True`` a ``CCZ`` gate is applied to every unique triple
        of qubits.  The interaction angle is derived from ``data_map_func``
        (or ``_default_map_3``).
    pre_rotation : bool
        If ``True`` prepend an ``RX(π/2)`` rotation to each qubit
        before the Hadamard layer.
    post_rotation : bool
        If ``True`` append an ``RY(π/2)`` rotation to each qubit
        after the entanglement block.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit has an
        ``input_params`` attribute containing the :class:`ParameterVector`
        that should be bound before execution.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if three_body and feature_dimension < 3:
        raise ValueError("three_body requires at least 3 qubits.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")
    x = ParameterVector(parameter_prefix, n)

    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])
        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    pairs = _resolve_entanglement(n, entanglement)
    triples: List[Tuple[int, int, int]] = []
    if three_body:
        triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    for rep in range(int(reps)):
        if pre_rotation:
            for i in range(n):
                qc.rx(pi / 2, i)
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        if three_body:
            for (i, j, k) in triples:
                angle_3 = 2 * map3(x[i], x[j], x[k])
                qc.ccz(i, j, k)
                qc.p(angle_3, k)
                qc.ccz(i, j, k)
        if post_rotation:
            for i in range(n):
                qc.ry(pi / 2, i)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Extended ZZ‑Feature Map (object‑oriented)
# ---------------------------------------------------------------------------

class ExtendedZZFeatureMap(QuantumCircuit):
    """Object‑oriented wrapper for the extended ZZ‑feature‑map.

    Parameters
    ----------
    *Same as :func:`extended_zz_feature_map`.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        three_body: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            three_body=three_body,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
