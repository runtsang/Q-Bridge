"""Extended ZZFeatureMap builder (Hadamard + ZZ entanglement via CX–P–CX, with optional higher‑order terms, rotations, and normalisation).

This module defines `zz_feature_map_extended` and `ZZFeatureMapExtended` that
extend the canonical ZZFeatureMap by:
* Optional third‑order (triplet) ZZ interactions implemented with Toffoli gates.
* Configurable pre‑ and post‑rotation layers.
* Normalisation toggle to scale raw data into the [0,π] range.
* Flexible entanglement specifications for pairwise and triplet couplings.
* Validation of parameter ranges and informative error messages.

The implementation remains Qiskit‑compatible, exposing an `input_params`
attribute for parameter binding and supporting the same data‑encoding workflow
as the original feature map.

Usage examples
--------------
>>> from zz_feature_map_extension import zz_feature_map_extended
>>> qc = zz_feature_map_extended(
...     feature_dimension=4,
...     reps=3,
...     entanglement='linear',
...     higher_order_entanglement='linear',
...     interaction_order=3,
...     pre_rotation=True,
...     post_rotation=True,
...     normalise_data=True,
... )
>>> qc.draw()
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
      - "linear": nearest neighbours (0,1), (1,2),...
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


def _resolve_triplet_entanglement(
    num_qubits: int,
    spec: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Return a list of three‑qubit tuples according to a spec.

    Supported specs:
      - "full": all distinct triples (i < j < k)
      - "linear": consecutive triples (0,1,2), (1,2,3),...
      - "circular": linear plus wrap‑around triples
      - explicit list of triples
      - callable: f(num_qubits) -> sequence of (i, j, k)
    """
    if isinstance(spec, str):
        if spec == "full":
            return [(i, j, k) for i in range(num_qubits)
                    for j in range(i + 1, num_qubits)
                    for k in range(j + 1, num_qubits)]
        if spec == "linear":
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        if spec == "circular":
            triples = [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
            if num_qubits > 3:
                triples.append((num_qubits - 2, num_qubits - 1, 0))
                triples.append((num_qubits - 1, 0, 1))
            return triples
        raise ValueError(f"Unknown higher_order_entanglement string: {spec!r}")

    if callable(spec):
        triples = list(spec(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in triples]

    # sequence of triples
    triples = [(int(i), int(j), int(k)) for (i, j, k) in spec]  # type: ignore[arg-type]
    for (i, j, k) in triples:
        if len({i, j, k})!= 3:
            raise ValueError("Higher‑order entanglement triples must contain distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits and 0 <= k < num_qubits):
            raise ValueError(
                f"Higher‑order entanglement triple {(i, j, k)} out of range for n={num_qubits}."
            )
    return triples


# ---------------------------------------------------------------------------
# Default mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Extended ZZFeatureMap (CX–P–CX for pairwise, CCX–P–CCX for triplets)
# ---------------------------------------------------------------------------

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    higher_order_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] | None = None,
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ-feature-map `QuantumCircuit`.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default=2
        Number of repetitions of the encoding block.
    entanglement : str or sequence or callable, default="full"
        Pairwise entanglement specification.
    higher_order_entanglement : str or sequence or callable, optional
        Triplet entanglement specification. Ignored if ``interaction_order`` <= 2.
    interaction_order : int, default=2
        Order of interactions to include (2 for pairwise, 3 for triplets).
    data_map_func : callable, optional
        User‑supplied mapping from feature vector to angle expression.
    parameter_prefix : str, default="x"
        Prefix for parameter vector names.
    insert_barriers : bool, default=False
        Insert barriers between logical blocks for visual clarity.
    pre_rotation : bool, default=False
        Apply an optional RY(π/2) rotation before the Hadamard layer.
    post_rotation : bool, default=False
        Apply an optional RZ(π/2) rotation after entanglement.
    normalise_data : bool, default=False
        Scale raw data to the [0, π] range before mapping.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A Qiskit circuit with an ``input_params`` attribute for parameter binding.

    Raises
    ------
    ValueError
        If parameters are out of range or invalid combinations are requested.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapExtended.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if interaction_order == 3 and not higher_order_entanglement:
        raise ValueError("higher_order_entanglement must be specified for interaction_order=3.")
    if interaction_order == 3 and feature_dimension < 3:
        raise ValueError("feature_dimension must be >= 3 for third‑order interactions.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Scaling helper
    scale = (lambda xi: pi * xi) if normalise_data else (lambda xi: xi)

    # Define mapping functions based on user input or defaults
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return scale(xi)

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - scale(xi)) * (pi - scale(xj))

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return (pi - scale(xi)) * (pi - scale(xj)) * (pi - scale(xk))
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Resolve triplet entanglement if needed
    triplets: List[Tuple[int, int, int]] = []
    if interaction_order == 3:
        triplets = _resolve_triplet_entanglement(n, higher_order_entanglement)  # type: ignore[arg-type]

    for rep in range(int(reps)):
        if pre_rotation:
            for q in range(n):
                qc.ry(pi / 2, q)
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        # Pairwise ZZ entanglers
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        # Triplet ZZ entanglers (if applicable)
        if interaction_order == 3:
            for (i, j, k) in triplets:
                angle_2 = 2 * map3(x[i], x[j], x[k])
                # CCX(i, j, k)
                qc.ccx(i, j, k)
                qc.p(angle_2, k)
                qc.ccx(i, j, k)
        if post_rotation:
            for q in range(n):
                qc.rz(pi / 2, q)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapExtended(QuantumCircuit):
    """Class‑style wrapper for the extended ZZFeatureMap.

    Parameters are identical to :func:`zz_feature_map_extended`.  The class
    composes the built circuit in place and exposes an ``input_params``
    attribute for parameter binding.

    Example
    -------
    >>> qc = ZZFeatureMapExtended(
   ...     feature_dimension=4,
   ...     reps=2,
   ...     interaction_order=3,
   ...     higher_order_entanglement='linear',
   ... )
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        higher_order_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] | None = None,
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        normalise_data: bool = False,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            higher_order_entanglement=higher_order_entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalise_data=normalise_data,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapExtended", "zz_feature_map_extended"]
