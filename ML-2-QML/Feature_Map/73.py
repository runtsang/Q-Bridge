"""SymmetricZZFeatureMap builder (Hadamard + ZZ entanglement via CX–P–CX with shared parameters and optional pre/post rotations).

This module implements a controlled modification of the canonical ZZ feature map.
The main differences are:
- Symmetrical pair couplings: the interaction angle for a pair (i, j) is identical to that for (j, i).
- Shared pair parameter: by default all pairs share a single parameter; this can be disabled to allow distinct angles per pair.
- Optional data scaling: a multiplicative factor can be applied to the raw input features before mapping.
- Optional pre‑ and post‑rotations: RY(π/2) and RZ(π/2) can be injected before the Hadamard layer and after the entanglement layer, respectively.
- The circuit keeps the same functional signature as the seed, so it can be used interchangeably in Qiskit workflows.

Supported datasets
------------------
Any dataset that can be represented as a real‑valued vector of length `feature_dimension` is supported.  The
feature vector is passed to the circuit via the `input_params` attribute and can be bound with
`circuit.bind_parameters(...)`.

Parameter constraints
--------------------
* `feature_dimension` must be an integer ≥ 2.
* `reps` must be a positive integer.
* `entanglement` must be one of the accepted specifications (`"full"`, `"linear"`, `"circular"` or an explicit list/callable).
* `data_map_func` must be a callable accepting a list of `ParameterExpression` and returning a `ParameterExpression`.
* `parameter_prefix` must be a string.
* `shared_pair_angle` must be a boolean.
* `data_scale` must be a real number.
* `pair_weight` must be a callable accepting two indices and returning a real number.

Example usage
-------------
>>> from qiskit import QuantumCircuit
>>> from mymodule import SymmetricZZFeatureMap
>>> qc = SymmetricZZFeatureMap(feature_dimension=3, reps=1, shared_pair_angle=True)
>>> qc.bind_parameters({p: 0.5 for p in qc.input_params})
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

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
      - "circular": linear plus wrap‑around (n-1,0) if n > 2
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
# Symmetric ZZFeatureMap
# ---------------------------------------------------------------------------

def symmetric_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    shared_pair_angle: bool = True,
    data_scale: float = 1.0,
    pair_weight: Callable[[int, int], float] | None = None,
    pre_rotation: bool = False,
    post_rotation: bool = False,
) -> QuantumCircuit:
    """Build a symmetric ZZ‑feature‑map with shared pair parameters and optional rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / input features.
    reps : int, default 2
        Number of repetitions (depth).
    entanglement : str or sequence or callable
        Specification of two‑qubit coupling pattern.
    data_map_func : callable, optional
        Function mapping a list of ParameterExpressions to a single
        ParameterExpression.  If None, default φ1 and φ2 are used.
    parameter_prefix : str, default "x"
        Prefix for the input parameters.
    insert_barriers : bool, default False
        Insert barriers between logical sections.
    name : str, optional
        Name of the circuit.
    shared_pair_angle : bool, default True
        Whether all pair couplings share a single parameter.
    data_scale : float, default 1.0
        Multiplicative scaling factor applied to raw input features.
    pair_weight : callable, optional
        Function ``(i, j) -> float`` returning a weight for a pair.
    pre_rotation : bool, default False
        Insert an RY(π/2) rotation before the Hadamard layer.
    post_rotation : bool, default False
        Insert an RZ(π/2) rotation after the entanglement layer.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    # --- Validation ---------------------------------------------------------
    if not isinstance(feature_dimension, int) or feature_dimension < 2:
        raise ValueError("feature_dimension must be an integer ≥ 2.")
    if not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be a positive integer.")
    if not isinstance(parameter_prefix, str):
        raise TypeError("parameter_prefix must be a string.")
    if not isinstance(shared_pair_angle, bool):
        raise TypeError("shared_pair_angle must be a boolean.")
    if not isinstance(data_scale, (int, float)):
        raise TypeError("data_scale must be numeric.")
    if pair_weight is not None and not callable(pair_weight):
        raise TypeError("pair_weight must be callable if provided.")
    if data_map_func is not None and not callable(data_map_func):
        raise TypeError("data_map_func must be callable if provided.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "SymmetricZZFeatureMap")

    # Input parameters
    x = ParameterVector(parameter_prefix, n)

    # Mapping functions
    if data_map_func is None:
        map1 = lambda xi: data_scale * _default_map_1(xi)
        map2 = lambda xi, xj: data_scale * _default_map_2(xi, xj)
    else:
        map1 = lambda xi: data_scale * data_map_func([xi])
        map2 = lambda xi, xj: data_scale * data_map_func([xi, xj])

    # Resolve entanglement pattern
    pairs = _resolve_entanglement(n, entanglement)
    num_pairs = len(pairs)

    # Pair parameters
    if shared_pair_angle:
        pair_params = ParameterVector("phi_pair", 1)
    else:
        pair_params = ParameterVector("phi_pair", num_pairs)

    # Default pair weight
    if pair_weight is None:
        def pair_weight(i: int, j: int) -> float:  # noqa: D401
            """Identity weight."""
            return 1.0

    # --- Build circuit ------------------------------------------------------
    for rep in range(int(reps)):
        if pre_rotation:
            qc.ry(pi / 2, range(n))
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        # ZZ via CX–P–CX
        for idx, (i, j) in enumerate(pairs):
            angle = 2 * map2(x[i], x[j]) * pair_weight(i, j)
            angle *= pair_params[0] if shared_pair_angle else pair_params[idx]
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if post_rotation:
            qc.rz(pi / 2, range(n))
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class SymmetricZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the symmetric ZZ‑feature‑map.

    Parameters are identical to :func:`symmetric_zz_feature_map`.  The ``input_params``
    attribute is preserved for compatibility with Qiskit's data‑encoding workflow.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "SymmetricZZFeatureMap",
        shared_pair_angle: bool = True,
        data_scale: float = 1.0,
        pair_weight: Callable[[int, int], float] | None = None,
        pre_rotation: bool = False,
        post_rotation: bool = False,
    ) -> None:
        built = symmetric_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            shared_pair_angle=shared_pair_angle,
            data_scale=data_scale,
            pair_weight=pair_weight,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["SymmetricZZFeatureMap", "symmetric_zz_feature_map"]
