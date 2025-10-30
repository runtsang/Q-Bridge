"""Extended ZZFeatureMap builder (Hadamard + ZZ entanglement via CX–P–CX, with optional 3‑body terms and pre/post rotations)."""

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
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2), …
      - "circular": linear + wrap‑around (n-1,0) if n > 2
      - explicit list of pairs
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
    entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Return a list of three‑qubit triples according to a simple entanglement spec.

    Supported specs:
      - "full": all combinations i<j<k
      - "linear": (0,1,2), (1,2,3), …
      - explicit list of triples
      - callable: f(num_qubits) -> sequence of (i, j, k)
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j, k) for i in range(num_qubits)
                    for j in range(i + 1, num_qubits)
                    for k in range(j + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        raise ValueError(f"Unknown triplet entanglement string: {entanglement!r}")

    if callable(entanglement):
        triples = list(entanglement(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in triples]

    triples = [(int(i), int(j), int(k)) for (i, j, k) in entanglement]  # type: ignore[arg-type]
    for (i, j, k) in triples:
        if len({i, j, k})!= 3:
            raise ValueError("Triplet entanglement must connect three distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits and 0 <= k < num_qubits):
            raise ValueError(f"Triplet {(i, j, k)} out of range for n={num_qubits}.")
    return triples


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
# Extended ZZFeatureMap (CX–P–CX for ZZ, optional 3‑body, pre/post rotations)
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    triplet_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map quantum circuit.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2 for two‑body
        interactions and >= 3 if triplet interactions are enabled.
    reps : int, default 2
        Number of repetitions of the basic block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit pairs for ZZ entanglers.
    triplet_entanglement : str | Sequence[Tuple[int, int, int]] | Callable
        Specification of three‑qubit triples for higher‑order ZZ entanglers.
        Ignored if `reps` is 0 or if `feature_dimension < 3`.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Custom mapping from feature vector to rotation angles.
        If None, defaults to the canonical two‑ and three‑body mappings.
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical layers.
    pre_rotation : bool, default False
        If True, apply an RX rotation with the mapped angle before the
        Hadamard and single‑qubit phase gates.
    post_rotation : bool, default False
        If True, append an RX rotation with the mapped angle after all
        entanglers in each repetition.
    normalize : bool, default False
        If True, scales all rotation angles to the range [0, 2π] by
        multiplying with 2π / (π * feature_dimension). This is useful
        when feature values are not in radians.
    name : str | None, default None
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit with an ``input_params`` attribute
        containing the ParameterVector.

    Raises
    ------
    ValueError
        If feature dimension is too small for the requested interactions,
        or if ``reps`` is not positive.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")
    if feature_dimension < 3 and triplet_entanglement not in (None, "none"):
        # triplet interactions require at least 3 qubits
        raise ValueError("feature_dimension must be >= 3 to enable triplet interactions.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    x = ParameterVector(parameter_prefix, n)

    # Resolve mapping functions
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

    two_pairs = _resolve_entanglement(n, entanglement)
    triplets = _resolve_triplet_entanglement(n, triplet_entanglement) if n >= 3 else []

    # Normalisation factor if requested
    norm_factor = (2 * pi) / (pi * n) if normalize else 1.0

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for i in range(n):
                qc.rx(2 * map1(x[i]) * norm_factor, i)

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]) * norm_factor, i)

        # Two‑body ZZ via CX–P–CX
        for (i, j) in two_pairs:
            angle_2 = 2 * map2(x[i], x[j]) * norm_factor
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Optional three‑body ZZ via CX–P–CX cascades
        for (i, j, k) in triplets:
            angle_3 = 2 * map3(x[i], x[j], x[k]) * norm_factor
            qc.cx(i, j)
            qc.cx(i, k)
            qc.p(angle_3, k)
            qc.cx(i, k)
            qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation:
            for i in range(n):
                qc.rx(2 * map1(x[i]) * norm_factor, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ExtendedZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Same as in :func:`extended_zz_feature_map`.
    reps : int, default 2
        Same as in :func:`extended_zz_feature_map`.
    entanglement : str | Sequence[Tuple[int, int]] | Callable, default "full"
        Same as in :func:`extended_zz_feature_map`.
    triplet_entanglement : str | Sequence[Tuple[int, int, int]] | Callable, default "full"
        Same as in :func:`extended_zz_feature_map`.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Same as in :func:`extended_zz_feature_map`.
    parameter_prefix : str, default "x"
        Same as in :func:`extended_zz_feature_map`.
    insert_barriers : bool, default False
        Same as in :func:`extended_zz_feature_map`.
    pre_rotation : bool, default False
        Same as in :func:`extended_zz_feature_map`.
    post_rotation : bool, default False
        Same as in :func:`extended_zz_feature_map`.
    normalize : bool, default False
        Same as in :func:`extended_zz_feature_map`.
    name : str, default "ExtendedZZFeatureMap"
        Name of the circuit instance.

    Notes
    -----
    The instance exposes an ``input_params`` attribute containing the
    ParameterVector for data binding, mirroring the functional interface.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        triplet_entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        normalize: bool = False,
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            triplet_entanglement=triplet_entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
