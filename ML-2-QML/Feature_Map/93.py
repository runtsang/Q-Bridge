"""ZZFeatureMapRZZExtension – a scalable, data‑encoding circuit for Qiskit."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
        - ``"full"``: all‑to‑all pairs (i < j)
        - ``"linear"``: nearest neighbors (0,1), (1,2),...
        - ``"circular"``: linear plus wrap‑around (n-1,0) if n > 2
        - explicit list of pairs like [(0, 2), (1, 3)]
        - callable: f(num_qubits) -> sequence of (i, j)

    Raises:
        ValueError: if an unknown spec or an invalid pair is supplied.
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


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# --------------------------------------------------------------------------- #
# Feature‑map construction
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    triple_scale: float = 1.0,
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a ZZ feature map with native RZZ two‑qubit entanglers and a third‑order CCPhase entangler.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 1.
    reps : int, default 2
        Number of repetitions of the encoding block. Must be > 0.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of qubit pairs to entangle. See ``_resolve_entanglement`` for details.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Optional custom mapping from a list of ParameterExpressions to a single ParameterExpression.
        When ``None`` the default polynomial maps are used.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        If ``True`` insert a barrier after each major sub‑step for easier circuit inspection.
    pair_scale : float, default 1.0
        Scaling factor applied to all pairwise RZZ angles. Must be > 0.
    triple_scale : float, default 1.0
        Scaling factor applied to all triple‑qubit CCPhase angles. Must be > 0.
    normalise_data : bool, default False
        If ``True`` each feature is multiplied by ``2π`` before mapping, ensuring values lie in [0, 2π].
    name : str | None, default None
        Circuit name. If ``None`` defaults to ``"ZZFeatureMapRZZExtension"``.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data binding via ``bind_parameters``.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    if pair_scale <= 0:
        raise ValueError("pair_scale must be > 0.")
    if triple_scale <= 0:
        raise ValueError("triple_scale must be > 0.")
    if triple_scale > 0 and feature_dimension < 3:
        raise ValueError("triple_scale > 0 requires at least 3 qubits for CCPhase entanglement.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtension")

    # Parameter vector for the raw features
    raw_params = ParameterVector(parameter_prefix, n)

    # Normalisation step
    if normalise_data:
        # Scale raw parameters to [0, 2π] before mapping
        norm_params = [2 * pi * p for p in raw_params]
    else:
        norm_params = list(raw_params)

    # Define mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        # data_map_func should accept a list of ParameterExpressions
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Build the circuit
    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(norm_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit RZZ entanglers
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(norm_params[i], norm_params[j]), i, j)

        if insert_barriers:
            qc.barrier()

        # Three‑qubit CCPhase entanglers (if applicable)
        if triple_scale > 0:
            # Generate all unique triples
            triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
            for (i, j, k) in triples:
                qc.ccphase(2 * triple_scale * map3(norm_params[i], norm_params[j], norm_params[k]), i, j, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach the input parameters for easy binding
    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZExtension(QuantumCircuit):
    """Class‑style wrapper for the extended RZZ feature map.

    The class behaves like a standard :class:`qiskit.circuit.QuantumCircuit` but
    pre‑configures the circuit with the parameters defined in
    :func:`zz_feature_map_rzz_extension`.  The ``input_params`` attribute is
    preserved for convenient parameter binding.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping function.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Insert barriers between sub‑steps if ``True``.
    pair_scale : float, default 1.0
        Scaling factor for pairwise RZZ gates.
    triple_scale : float, default 1.0
        Scaling factor for three‑qubit CCPhase gates.
    normalise_data : bool, default False
        Normalise raw features to [0, 2π] if ``True``.
    name : str, default "ZZFeatureMapRZZExtension"
        Circuit name.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        triple_scale: float = 1.0,
        normalise_data: bool = False,
        name: str = "ZZFeatureMapRZZExtension",
    ) -> None:
        built = zz_feature_map_rzz_extension(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            triple_scale,
            normalise_data,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtension", "zz_feature_map_rzz_extension"]
