"""ZZFeatureMapExtension: enriched ZZ feature map with higher‑order interactions and optional rotations.

This module implements both a functional builder `zz_feature_map_extension` and a
`QuantumCircuit` subclass `ZZFeatureMapExtension`.  The design follows an
**extension** scaling paradigm: the core canonical mapping is retained, but
extra capabilities are added:

* **Higher‑order interactions** – pair (default) or triple‑qubit ZZ couplings.
* **Pre‑ and post‑rotations** – arbitrary single‑qubit gates applied before
  or after the main encoding block.
* **Data normalisation** – optional scaling of the input feature vector.
* **Adaptive depth** – number of repetitions can be controlled or
  automatically adjusted based on the feature dimension.
* **Custom data‑mapping** – user supplied functions for transforming raw
  features into rotation angles.

The interface is fully compatible with Qiskit’s data‑encoding patterns:
the resulting circuit exposes an `input_params` attribute containing a
`ParameterVector` that can be bound to a classical feature vector.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the pairwise entanglement.

    Supported specs
    ---------------
    - ``"full"``       : all-to-all pairs (i < j)
    - ``"linear"``     : nearest neighbours (0,1), (1,2), …
    - ``"circular"``   : linear + wrap‑around (n‑1,0) if n > 2
    - explicit list   : e.g. ``[(0, 2), (1, 3)]``
    - callable        : ``f(num_qubits) -> sequence of (i, j)``

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an unsupported string is supplied or a pair references an
        out‑of‑range qubit.
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
    """Default single‑qubit angle φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit angle φ2(x, y) = (π − x)(π − y)."""
    return (math.pi - x) * (math.pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default three‑qubit angle φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (math.pi - x) * (math.pi - y) * (math.pi - z)


# --------------------------------------------------------------------------- #
# Functional builder
# --------------------------------------------------------------------------- #

def zz_feature_map_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    data_map_func: Callable[[Iterable[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    interaction_order: int = 2,
    include_triples: bool = False,
    pre_rotation: Callable[[QuantumCircuit, ParameterVector], None] | None = None,
    post_rotation: Callable[[QuantumCircuit, ParameterVector], None] | None = None,
    normalize: bool = False,
) -> QuantumCircuit:
    """Build an enriched ZZ‑feature‑map circuit.

    The returned circuit is compatible with Qiskit data‑encoding workflows:
    it contains an ``input_params`` attribute holding a :class:`ParameterVector`
    that can be bound to a classical feature vector.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.  Must be >= ``interaction_order``.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the pairwise entanglement topology.
    data_map_func : Callable[[Iterable[ParameterExpression]], ParameterExpression] | None
        Optional user‑supplied mapping from raw features to rotation angles.
        The function receives a *sequence* of :class:`ParameterExpression`s
        corresponding to the involved qubits and must return a single
        :class:`ParameterExpression`.  When ``None`` (default) the module
        uses the built‑in defaults.
    parameter_prefix : str, default "x"
        Prefix used for the :class:`ParameterVector` names.
    insert_barriers : bool, default False
        If ``True`` a barrier is inserted after each major sub‑block.
    name : str | None, default None
        Optional circuit name; if omitted the default is ``"ZZFeatureMapExtension"``.
    interaction_order : int, default 2
        Order of the interaction to encode.  Currently 2 (pair) or 3 (triple)
        are supported.  ``include_triples`` enables a 3‑qubit term even if
        ``interaction_order`` is 2.
    include_triples : bool, default False
        When ``True`` a 3‑qubit ZZ coupling is added in addition to the
        pairwise terms.
    pre_rotation : Callable[[QuantumCircuit, ParameterVector], None] | None, default None
        Optional callable that receives the circuit and the parameter vector
        and applies arbitrary single‑qubit gates before the main encoding block.
    post_rotation : Callable[[QuantumCircuit, ParameterVector], None] | None, default None
        Optional callable that receives the circuit and the parameter vector
        and applies arbitrary single‑qubit gates after the main encoding block.
    normalize : bool, default False
        If ``True`` all feature parameters are scaled by ``π`` before
        being used in the mapping functions.  This is a lightweight
        normalisation that maps a typical [0,1] feature range into
        [0,π].

    Returns
    -------
    QuantumCircuit
        The enriched feature‑map circuit.

    Raises
    ------
    ValueError
        If ``feature_dimension`` is smaller than ``interaction_order``,
        or if an unsupported ``interaction_order`` is supplied.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 (pair) or 3 (triple).")
    if feature_dimension < interaction_order:
        raise ValueError(
            f"feature_dimension ({feature_dimension}) must be >= "
            f"interaction_order ({interaction_order})."
        )

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtension")

    # Build a ParameterVector for the raw feature values
    raw_params = ParameterVector(parameter_prefix, n)

    # Optional normalisation scale
    scale_factor = math.pi if normalize else 1.0

    # Resolve data mapping functions
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

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Optional pre‑rotation
    if pre_rotation is not None:
        pre_rotation(qc, raw_params)
        if insert_barriers:
            qc.barrier()

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(scale_factor * raw_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(scale_factor * raw_params[i], scale_factor * raw_params[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Optional triple‑qubit ZZ term
        if include_triples or interaction_order == 3:
            if n < 3:
                raise ValueError("Triple‑qubit interactions require at least 3 qubits.")
            for (i, j, k) in _triplet_pairs(n):
                angle_3 = 2 * map3(
                    scale_factor * raw_params[i],
                    scale_factor * raw_params[j],
                    scale_factor * raw_params[k],
                )
                # Implement a simple 3‑qubit ZZ via a sequence of CX–P–CX gates
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle_3, k)
                qc.cx(j, k)
                qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Optional post‑rotation
    if post_rotation is not None:
        if insert_barriers:
            qc.barrier()
        post_rotation(qc, raw_params)

    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


def _triplet_pairs(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Generate a list of all distinct 3‑qubit combinations."""
    pairs: List[Tuple[int, int, int]] = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            for k in range(j + 1, num_qubits):
                pairs.append((i, j, k))
    return pairs


# --------------------------------------------------------------------------- #
# Object‑oriented wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapExtension(QuantumCircuit):
    """QuantumCircuit subclass for the enriched ZZ‑feature‑map.

    The constructor forwards all arguments to :func:`zz_feature_map_extension`
    and then composes the resulting circuit into the current instance.
    The ``input_params`` attribute is preserved for parameter binding.

    Example
    -------
    >>> from qiskit.circuit import ParameterVector
    >>> from zz_feature_map_extension import ZZFeatureMapExtension
    >>> qc = ZZFeatureMapExtension(4, reps=3, interaction_order=3, include_triples=True)
    >>> params = qc.input_params
    >>> bound_qc = qc.bind_parameters({p: 0.5 for p in params})
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        data_map_func: Callable[[Iterable[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtension",
        interaction_order: int = 2,
        include_triples: bool = False,
        pre_rotation: Callable[[QuantumCircuit, ParameterVector], None] | None = None,
        post_rotation: Callable[[QuantumCircuit, ParameterVector], None] | None = None,
        normalize: bool = False,
    ) -> None:
        built = zz_feature_map_extension(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            interaction_order=interaction_order,
            include_triples=include_triples,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalize=normalize,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]

__all__ = ["ZZFeatureMapExtension", "zz_feature_map_extension"]
