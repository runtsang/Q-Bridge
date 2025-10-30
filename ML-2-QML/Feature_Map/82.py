"""Controlled‑modification ZZFeatureMap: shared data‑derived entanglement phase.

The circuit follows the canonical ZZFeatureMap structure but replaces the per‑pair
entanglement phase with a single shared phase derived from the entire data
vector.  This enforces symmetry, reduces the number of independent parameters,
and can improve expressivity for certain datasets.

Key features
------------
* **Shared entanglement phase** – a single data‑derived parameter controls all
  ZZ couplers.
* **Optional data normalisation** – scale input features to [0, π] before mapping.
* **Custom entanglement patterns** – full, linear, circular, explicit list,
  or callable.
* **Barrier insertion** – useful for visualisation and debugging.
* **Data mapping functions** – override default φ₁ and φ₂ or provide a custom
  shared‑angle function.
* **Parameter binding** – the returned circuit exposes an ``input_params`` attribute
  compatible with Qiskit's data‑encoding workflows.

Usage
-----
>>> from zz_feature_map_controlled import zz_feature_map_controlled, ZZFeatureMapControlled
>>> qc = zz_feature_map_controlled(feature_dimension=4, reps=1)
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
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
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


def _default_shared_angle_func(x: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default shared angle derived from the sum of all feature parameters."""
    return sum(x)


# ---------------------------------------------------------------------------
# Controlled‑modification ZZFeatureMap
# ---------------------------------------------------------------------------

def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    shared_angle_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    shared_angle_prefix: str = "s",
    normalize_data: bool = True,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a controlled‑modification ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the classical feature vector.
        Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str or sequence or callable, default "full"
        Specification of which qubit pairs receive the shared entanglement phase.
    data_map_func : callable, optional
        Function that maps a list of ParameterExpressions to a single
        ParameterExpression.  If None, defaults to the product of the two
        inputs (see :func:`_default_map_2`).
    shared_angle_func : callable, optional
        Function that maps the full list of feature parameters to a single
        ParameterExpression used for all pair entanglers.  If None, defaults
        to the sum of the parameters.
    parameter_prefix : str, default "x"
        Prefix for the data parameter vector.
    shared_angle_prefix : str, default "s"
        Prefix for the shared‑angle parameter vector (used only for naming
        purposes; the shared angle is derived from the data).
    normalize_data : bool, default True
        If True, multiply each feature by π before mapping, ensuring the
        parameter space is [0, π].
    insert_barriers : bool, default False
        If True, insert barriers between major blocks for readability.
    name : str, optional
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit exposes an
        ``input_params`` attribute containing the ParameterVector for the
        data features.

    Raises
    ------
    ValueError
        If `feature_dimension` < 2, `reps` <= 0, or invalid entanglement spec.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapControlled.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Data parameters
    x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Shared‑angle function
    if shared_angle_func is None:
        shared_func = _default_shared_angle_func
    else:
        shared_func = shared_angle_func

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            param = map1(x[i])
            if normalize_data:
                param = param * pi
            qc.p(2 * param, i)

        if insert_barriers:
            qc.barrier()

        # Shared entanglement phase
        shared_angle = 2 * shared_func(x)
        for (i, j) in pairs:
            # CX–P–CX implementation of a shared‑phase ZZ gate
            qc.cx(i, j)
            qc.p(shared_angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification ZZ‑feature‑map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        shared_angle_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        shared_angle_prefix: str = "s",
        normalize_data: bool = True,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            shared_angle_func=shared_angle_func,
            parameter_prefix=parameter_prefix,
            shared_angle_prefix=shared_angle_prefix,
            normalize_data=normalize_data,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled"]
